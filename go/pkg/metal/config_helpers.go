// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

func firstPositiveInt(values ...int) int {
	return FirstPositiveInt(values...)
}

// FirstPositiveInt returns the first positive value from values, or zero.
// Model packages use it while normalising nested config.json shapes.
func FirstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func firstNonEmptyString(values ...string) string {
	return FirstNonEmptyString(values...)
}

// FirstNonEmptyString returns the first non-empty value from values, or "".
// Model packages use it while normalising aliases and nested text configs.
func FirstNonEmptyString(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
