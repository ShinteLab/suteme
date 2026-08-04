package training

import (
	"testing"

	"github.com/ShinteLab/suteme"
)

func sample(label int, vals ...float64) suteme.TrainingSample {
	in := make([]float64, suteme.InputSize)
	copy(in, vals)
	return suteme.TrainingSample{Input: in, Label: label}
}

// 同じ入力が何度渡されても1件に畳まれること（学習を複数回押しても増えない）
func TestMergeSamplesRemovesDuplicates(t *testing.T) {
	a := []suteme.TrainingSample{sample(0, 1), sample(1, 2), sample(2, 3)}

	got := MergeSamples(a, a)
	if len(got) != 3 {
		t.Fatalf("重複が残っている: %d件, want 3", len(got))
	}
	// もう一度同じものを足しても増えない
	if got = MergeSamples(got, a, a); len(got) != 3 {
		t.Fatalf("重複が残っている: %d件, want 3", len(got))
	}
}

// 同じ入力に別のラベルが付いた場合は後勝ち（ラベルの付け直しが反映される）
func TestMergeSamplesLastLabelWins(t *testing.T) {
	old := []suteme.TrainingSample{sample(0, 1, 2, 3)}
	fixed := []suteme.TrainingSample{sample(7, 1, 2, 3)}

	got := MergeSamples(old, fixed)
	if len(got) != 1 {
		t.Fatalf("件数 = %d, want 1", len(got))
	}
	if got[0].Label != 7 {
		t.Errorf("ラベル = %d, want 7（後勝ち）", got[0].Label)
	}
}

// 異なる入力は畳まれず、最初に現れた順序が保たれること
func TestMergeSamplesKeepsDistinctAndOrder(t *testing.T) {
	a := []suteme.TrainingSample{sample(0, 1), sample(1, 2)}
	b := []suteme.TrainingSample{sample(2, 3), sample(0, 1)}

	got := MergeSamples(a, b)
	if len(got) != 3 {
		t.Fatalf("件数 = %d, want 3", len(got))
	}
	want := []int{0, 1, 2}
	for i, w := range want {
		if got[i].Label != w {
			t.Errorf("got[%d].Label = %d, want %d（順序が保たれていない）", i, got[i].Label, w)
		}
	}
}

func TestMergeSamplesEmpty(t *testing.T) {
	if got := MergeSamples(); len(got) != 0 {
		t.Errorf("件数 = %d, want 0", len(got))
	}
	if got := MergeSamples(nil, nil); len(got) != 0 {
		t.Errorf("件数 = %d, want 0", len(got))
	}
}
