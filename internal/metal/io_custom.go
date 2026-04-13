//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "mlx/c/mlx.h"

// Forward declarations for Go callback bridges.
// These signatures must match what cgo generates for the //export functions.
extern _Bool goIOIsOpen(void* desc);
extern _Bool goIOGood(void* desc);
extern size_t goIOTell(void* desc);
extern void  goIOSeek(void* desc, int64_t off, int whence);
extern void  goIORead(void* desc, char* data, size_t n);
extern void  goIOReadAtOffset(void* desc, char* data, size_t n, size_t off);
extern void  goIOWrite(void* desc, char* data, size_t n);
extern char* goIOLabel(void* desc);
extern void  goIOFree(void* desc);

// Thin wrappers to bridge const qualifiers between the mlx_io_vtable
// signatures and the Go-exported callbacks (cgo drops const).
static bool wrap_is_open(void* d) { return goIOIsOpen(d); }
static bool wrap_good(void* d) { return goIOGood(d); }
static size_t wrap_tell(void* d) { return goIOTell(d); }
static void wrap_seek(void* d, int64_t off, int w) { goIOSeek(d, off, w); }
static void wrap_read(void* d, char* data, size_t n) { goIORead(d, data, n); }
static void wrap_read_at_offset(void* d, char* data, size_t n, size_t off) { goIOReadAtOffset(d, data, n, off); }
static void wrap_write(void* d, const char* data, size_t n) { goIOWrite(d, (char*)data, n); }
static const char* wrap_label(void* d) { return (const char*)goIOLabel(d); }
static void wrap_free(void* d) { goIOFree(d); }

// Build the vtable once in C using the wrapper functions.
static mlx_io_vtable go_io_vtable = {
    &wrap_is_open,
    &wrap_good,
    &wrap_tell,
    &wrap_seek,
    &wrap_read,
    &wrap_read_at_offset,
    &wrap_write,
    &wrap_label,
    &wrap_free,
};

// Accepts uintptr_t to avoid Go unsafe.Pointer conversion from integer.
static mlx_io_reader make_go_reader(uintptr_t id) {
    return mlx_io_reader_new((void*)id, go_io_vtable);
}

// Accepts uintptr_t to avoid Go unsafe.Pointer conversion from integer.
static mlx_io_writer make_go_writer(uintptr_t id) {
    return mlx_io_writer_new((void*)id, go_io_vtable);
}
*/
import "C"

import (
	"io"
	"iter"
	"runtime"
	"sync"
	"unsafe"

	"dappco.re/go/core"
)

// ioStream is the Go-side descriptor passed through C void* callbacks.
// It wraps a Go io.ReadWriteSeeker to satisfy the mlx_io_vtable contract.
type ioStream struct {
	rws   io.ReadWriteSeeker
	label string
	size  int64 // total size, required for SEEK_END
	good  bool
	open  bool
}

// ioRegistry maps C void* descriptor pointers back to Go ioStream values.
// This avoids passing Go pointers through C (which violates cgo pointer rules).
var (
	ioRegistryMu sync.Mutex
	ioRegistry   = make(map[uintptr]*ioStream)
	ioNextID     uintptr
)

func registerIOStream(s *ioStream) uintptr {
	ioRegistryMu.Lock()
	defer ioRegistryMu.Unlock()
	ioNextID++
	id := ioNextID
	ioRegistry[id] = s
	return id
}

func lookupIOStream(desc unsafe.Pointer) *ioStream {
	ioRegistryMu.Lock()
	defer ioRegistryMu.Unlock()
	return ioRegistry[uintptr(desc)]
}

func unregisterIOStream(desc unsafe.Pointer) {
	ioRegistryMu.Lock()
	defer ioRegistryMu.Unlock()
	delete(ioRegistry, uintptr(desc))
}

// --- C callback exports (called from MLX-C via the vtable) ---

//export goIOIsOpen
func goIOIsOpen(desc unsafe.Pointer) C.bool {
	s := lookupIOStream(desc)
	if s == nil {
		return C.bool(false)
	}
	return C.bool(s.open)
}

//export goIOGood
func goIOGood(desc unsafe.Pointer) C.bool {
	s := lookupIOStream(desc)
	if s == nil {
		return C.bool(false)
	}
	return C.bool(s.good)
}

//export goIOTell
func goIOTell(desc unsafe.Pointer) C.size_t {
	s := lookupIOStream(desc)
	if s == nil {
		return 0
	}
	pos, err := s.rws.Seek(0, io.SeekCurrent)
	if err != nil {
		s.good = false
		return 0
	}
	return C.size_t(pos)
}

//export goIOSeek
func goIOSeek(desc unsafe.Pointer, off C.int64_t, whence C.int) {
	s := lookupIOStream(desc)
	if s == nil {
		return
	}
	var goWhence int
	switch whence {
	case 0: // SEEK_SET
		goWhence = io.SeekStart
	case 1: // SEEK_CUR
		goWhence = io.SeekCurrent
	case 2: // SEEK_END
		goWhence = io.SeekEnd
	default:
		s.good = false
		return
	}
	_, err := s.rws.Seek(int64(off), goWhence)
	if err != nil {
		s.good = false
	}
}

//export goIORead
func goIORead(desc unsafe.Pointer, data *C.char, n C.size_t) {
	s := lookupIOStream(desc)
	if s == nil {
		return
	}
	buf := unsafe.Slice((*byte)(unsafe.Pointer(data)), int(n))
	_, err := io.ReadFull(s.rws, buf)
	if err != nil {
		s.good = false
	}
}

//export goIOReadAtOffset
func goIOReadAtOffset(desc unsafe.Pointer, data *C.char, n C.size_t, off C.size_t) {
	s := lookupIOStream(desc)
	if s == nil {
		return
	}
	_, err := s.rws.Seek(int64(off), io.SeekStart)
	if err != nil {
		s.good = false
		return
	}
	buf := unsafe.Slice((*byte)(unsafe.Pointer(data)), int(n))
	_, err = io.ReadFull(s.rws, buf)
	if err != nil {
		s.good = false
	}
}

//export goIOWrite
func goIOWrite(desc unsafe.Pointer, data *C.char, n C.size_t) {
	s := lookupIOStream(desc)
	if s == nil {
		return
	}
	buf := unsafe.Slice((*byte)(unsafe.Pointer(data)), int(n))
	_, err := s.rws.Write(buf)
	if err != nil {
		s.good = false
	}
}

//export goIOLabel
func goIOLabel(desc unsafe.Pointer) *C.char {
	s := lookupIOStream(desc)
	if s == nil {
		return C.CString("<unknown go stream>")
	}
	// MLX-C does not free this; it only reads it transiently.
	// We return a C string that lives until the stream is freed.
	return C.CString(s.label)
}

//export goIOFree
func goIOFree(desc unsafe.Pointer) {
	unregisterIOStream(desc)
}

// LoadSafetensorsFromReader loads tensors from a custom io.ReadWriteSeeker,
// returning an iterator over (name, array) pairs. This enables loading
// safetensors from in-memory buffers, network streams, or encrypted storage
// without touching disk.
//
//	buf := bytes.NewReader(safetensorsBytes)
//	rws := io.NewSectionReader(buf, 0, int64(len(safetensorsBytes)))
//	for name, arr := range metal.LoadSafetensorsFromReader(rws, int64(len(safetensorsBytes)), "memory") {
//	    weights[name] = arr
//	}
func LoadSafetensorsFromReader(rws io.ReadWriteSeeker, size int64, label string) iter.Seq2[string, *Array] {
	Init()
	return func(yield func(string, *Array) bool) {
		stream := &ioStream{
			rws:   rws,
			label: label,
			size:  size,
			good:  true,
			open:  true,
		}
		id := registerIOStream(stream)
		reader := C.make_go_reader(C.uintptr_t(id))
		defer C.mlx_io_reader_free(reader)

		string2array := C.mlx_map_string_to_array_new()
		defer C.mlx_map_string_to_array_free(string2array)

		string2string := C.mlx_map_string_to_string_new()
		defer C.mlx_map_string_to_string_free(string2string)

		cpu := C.mlx_default_cpu_stream_new()
		defer C.mlx_stream_free(cpu)

		rc := C.mlx_load_safetensors_reader(&string2array, &string2string, reader, cpu)
		if rc != 0 {
			return
		}

		it := C.mlx_map_string_to_array_iterator_new(string2array)
		defer C.mlx_map_string_to_array_iterator_free(it)

		for {
			var key *C.char
			value := C.mlx_array_new()
			if C.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 {
				break
			}

			name := C.GoString(key)
			arr := &Array{ctx: value, name: name}
			runtime.SetFinalizer(arr, finalizeArray)
			if !yield(name, arr) {
				break
			}
		}
	}
}

// LoadAllSafetensorsFromReader loads all tensors from a custom reader into a map.
// Returns an error if the data cannot be parsed.
//
//	weights, err := metal.LoadAllSafetensorsFromReader(rws, size, "memory")
func LoadAllSafetensorsFromReader(rws io.ReadWriteSeeker, size int64, label string) (map[string]*Array, error) {
	tensors := make(map[string]*Array)
	for name, arr := range LoadSafetensorsFromReader(rws, size, label) {
		tensors[name] = arr
	}
	if len(tensors) == 0 {
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.LoadAllSafetensorsFromReader", "no tensors loaded from custom reader", nil)
	}
	return tensors, nil
}

// SaveSafetensorsToWriter writes tensors to a custom io.ReadWriteSeeker.
// The metadata map is optional (pass nil for no metadata).
// This enables saving safetensors to in-memory buffers, network streams,
// or encrypted storage without touching disk.
//
//	var buf bytes.Buffer
//	rws := newBytesRWS(&buf)
//	err := metal.SaveSafetensorsToWriter(rws, 0, "memory", weights, nil)
func SaveSafetensorsToWriter(rws io.ReadWriteSeeker, size int64, label string, tensors map[string]*Array, metadata map[string]string) error {
	Init()
	stream := &ioStream{
		rws:   rws,
		label: label,
		size:  size,
		good:  true,
		open:  true,
	}
	id := registerIOStream(stream)
	writer := C.make_go_writer(C.uintptr_t(id))
	defer C.mlx_io_writer_free(writer)

	string2array := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(string2array)

	for name, arr := range tensors {
		cName := C.CString(name)
		C.mlx_map_string_to_array_insert(string2array, cName, arr.ctx)
		C.free(unsafe.Pointer(cName))
	}

	string2string := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_string_free(string2string)

	if metadata != nil {
		for k, v := range metadata {
			cK := C.CString(k)
			cV := C.CString(v)
			C.mlx_map_string_to_string_insert(string2string, cK, cV)
			C.free(unsafe.Pointer(cK))
			C.free(unsafe.Pointer(cV))
		}
	}

	rc := C.mlx_save_safetensors_writer(writer, string2array, string2string)
	if rc != 0 {
		if err := lastError(); err != nil {
			return err
		}
		return core.E("mlx.SaveSafetensorsToWriter", "save failed", nil)
	}
	return nil
}

// MapGet retrieves an array from a string-to-array map by key.
// Returns the array and true if found, nil and false otherwise.
//
//	arr, ok := metal.MapGet(tensorMap, "model.layers.0.self_attn.q_proj.weight")
func MapGet(m C.mlx_map_string_to_array, key string) (*Array, bool) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	value := C.mlx_array_new()
	rc := C.mlx_map_string_to_array_get(&value, m, cKey)
	if rc != 0 {
		C.mlx_array_free(value)
		return nil, false
	}

	arr := &Array{ctx: value, name: key}
	runtime.SetFinalizer(arr, finalizeArray)
	return arr, true
}
