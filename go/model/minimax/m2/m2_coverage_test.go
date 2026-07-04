// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"
)

// These tests close the remaining statement-coverage gaps in m2.go. Both are
// pure white-box helper legs — no IO, no device, no model (AX-11).

func TestM2Cover_attentionSpec_DefaultRoleShape(t *testing.T) {
	// attentionSpec selects its weight shape by attention role (Q / K-V / O);
	// any other role falls through to the {hidden, hidden} default. Drive that
	// default by asking for a non-attention role so the fallback shape is used.
	plan := miniMaxM2SmallJANGTQPlan(t) // HiddenSize 2
	spec := plan.attentionSpec(0, "q_proj", TensorRoleRouterGate)
	if !sameUint64Slice(spec.Shape, []uint64{2, 2}) {
		t.Fatalf("default-role shape = %+v, want {hidden, hidden} = [2 2]", spec.Shape)
	}
	if spec.Role != TensorRoleRouterGate {
		t.Fatalf("role = %v, want the passed-through role", spec.Role)
	}
}

func TestM2Cover_cloneJANGQuantizationInfo_Nil(t *testing.T) {
	// A nil quantization info clones to nil rather than dereferencing.
	if got := cloneJANGQuantizationInfo(nil); got != nil {
		t.Fatalf("cloneJANGQuantizationInfo(nil) = %+v, want nil", got)
	}
}
