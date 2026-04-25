// AX-6-exception: runtime import scoped here so consumers can call mlx.GC() instead of runtime.GC() directly.
package metal

import "runtime"

func RuntimeGC() { runtime.GC() }
