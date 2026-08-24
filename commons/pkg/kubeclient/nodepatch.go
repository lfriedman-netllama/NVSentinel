// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kubeclient

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	typedcorev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/util/retry"
)

// NodePatcher applies Node mutations with cache-aware retries.
//
// The zero value is ready to use.
type NodePatcher struct {
	pendingVersions sync.Map // map[string]string
}

// Patch applies mutate to a copy of cached and writes only the resulting diff.
// cached may be nil when an informer is unavailable; in that case Patch reads the
// current Node from the API server. A write not yet observed at cached's
// ResourceVersion is also refreshed before the next mutation.
func (p *NodePatcher) Patch(
	ctx context.Context,
	nodes typedcorev1.NodeInterface,
	nodeName string,
	cached *v1.Node,
	mutate func(*v1.Node) error,
) (bool, error) {
	current, err := p.currentNode(ctx, nodes, nodeName, cached)
	if err != nil {
		return false, err
	}

	changed := false

	err = retry.OnError(nodePatchBackoff(), isRetryableNodePatchError, func() error {
		desired := current.DeepCopy()

		if err := mutate(desired); err != nil {
			return fmt.Errorf("mutate node %q: %w", nodeName, err)
		}

		patch, err := NodeMergePatch(current, desired)
		if err != nil {
			return fmt.Errorf("build merge patch for node %q: %w", nodeName, err)
		}

		if patch == nil {
			return nil
		}

		updated, err := nodes.Patch(ctx, nodeName, types.MergePatchType, patch, metav1.PatchOptions{})
		if err == nil {
			p.pendingVersions.Store(nodeName, updated.ResourceVersion)

			changed = true

			return nil
		}

		if errors.IsConflict(err) {
			patchErr := err

			current, err = nodes.Get(ctx, nodeName, metav1.GetOptions{})
			if err != nil {
				return fmt.Errorf("refresh node %q after patch conflict: %w", nodeName, err)
			}

			return fmt.Errorf("patch node %q: %w", nodeName, patchErr)
		}

		return fmt.Errorf("patch node %q: %w", nodeName, err)
	})
	if err != nil {
		return false, err
	}

	return changed, nil
}

func (p *NodePatcher) currentNode(
	ctx context.Context,
	nodes typedcorev1.NodeInterface,
	nodeName string,
	cached *v1.Node,
) (*v1.Node, error) {
	writtenVersionValue, hasPendingWrite := p.pendingVersions.Load(nodeName)
	if !hasPendingWrite {
		if cached != nil {
			return cached, nil
		}

		current, err := nodes.Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("get node %q from API server: %w", nodeName, err)
		}

		return current, nil
	}

	writtenVersion, _ := writtenVersionValue.(string)
	if cached != nil && writtenVersion != "" && cached.ResourceVersion == writtenVersion {
		p.pendingVersions.CompareAndDelete(nodeName, writtenVersionValue)

		return cached, nil
	}

	current, err := nodes.Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("refresh node %q while pending write is not in cache: %w", nodeName, err)
	}

	p.pendingVersions.CompareAndDelete(nodeName, writtenVersionValue)

	return current, nil
}

func nodePatchBackoff() wait.Backoff {
	return wait.Backoff{
		Steps:    10,
		Duration: 20 * time.Millisecond,
		Factor:   2,
		Jitter:   0.1,
	}
}

func isRetryableNodePatchError(err error) bool {
	return errors.IsConflict(err) ||
		errors.IsServerTimeout(err) ||
		errors.IsTooManyRequests(err) ||
		errors.IsTimeout(err) ||
		errors.IsServiceUnavailable(err)
}

// NodeMergePatch builds an RFC 7386 JSON merge patch carrying differences in labels,
// annotations, taints, and unschedulable state. It returns a nil patch when the two
// nodes already agree, so callers can skip a no-op write.
//
// The patch is assembled key by key rather than by marshalling modified, because
// marshalling a Node emits every populated field. A caller that read original from an
// informer cache holding a projected Node would then patch the projection's gaps back
// over the real object. Emitting only keys that differ means fields missing from both
// sides are left untouched.
//
// Taints are emitted only when the caller changed them. A projected Node whose Spec
// is empty on both sides therefore cannot erase taints from the real object.
func NodeMergePatch(original, modified *v1.Node) ([]byte, error) {
	root := map[string]any{}
	metadata := map[string]any{}

	if labels := stringMapMergePatch(original.Labels, modified.Labels); labels != nil {
		metadata["labels"] = labels
	}

	if annotations := stringMapMergePatch(original.Annotations, modified.Annotations); annotations != nil {
		metadata["annotations"] = annotations
	}

	if len(metadata) > 0 {
		root["metadata"] = metadata
	}

	spec := map[string]any{}
	if !taintsEqual(original.Spec.Taints, modified.Spec.Taints) {
		spec["taints"] = modified.Spec.Taints
	}

	if original.Spec.Unschedulable != modified.Spec.Unschedulable {
		spec["unschedulable"] = modified.Spec.Unschedulable
	}

	if len(spec) > 0 {
		root["spec"] = spec
	}

	if len(root) == 0 {
		return nil, nil
	}

	patch, err := json.Marshal(root)
	if err != nil {
		return nil, fmt.Errorf("marshal merge patch for node %s: %w", original.Name, err)
	}

	return patch, nil
}

func taintsEqual(original, modified []v1.Taint) bool {
	if len(original) != len(modified) {
		return false
	}

	for idx := range original {
		if !reflect.DeepEqual(original[idx], modified[idx]) {
			return false
		}
	}

	return true
}

// stringMapMergePatch returns the merge patch entries that turn original into
// modified: added and changed keys map to their new value, removed keys map to nil so
// the API server deletes them. It returns nil when the two maps already agree.
func stringMapMergePatch(original, modified map[string]string) map[string]any {
	patch := map[string]any{}

	for key, value := range modified {
		if current, exists := original[key]; !exists || current != value {
			patch[key] = value
		}
	}

	for key := range original {
		if _, exists := modified[key]; !exists {
			patch[key] = nil
		}
	}

	if len(patch) == 0 {
		return nil
	}

	return patch
}
