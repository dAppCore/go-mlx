// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

func TestDefaultLemmaNewSessionText_Good(t *testing.T) {
	if !core.Contains(DefaultLemmaNewSessionText, "Lemma") || !core.Contains(DefaultLemmaNewSessionText, "Lethean Model Engine") {
		t.Fatalf("DefaultLemmaNewSessionText = %q, want Lemma engine default", DefaultLemmaNewSessionText)
	}
	if DefaultNewSessionText != DefaultLemmaNewSessionText {
		t.Fatalf("DefaultNewSessionText = %q, want Lemma default alias", DefaultNewSessionText)
	}
}
