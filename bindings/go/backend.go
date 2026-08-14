package stridealign

// BackendRecord describes a SIMD backend compiled into the package.
type BackendRecord struct {
	Kind      string
	Name      string
	Compiled  bool
	Available bool
}

// Backend reports the selected native backend.
func Backend() string { return SIMDLevel() }

// AvailableBackends reports the compile-time backend. Go build tags select one
// target, so unlike R's fat package this list contains exactly one entry.
func AvailableBackends() []BackendRecord {
	name := Backend()
	return []BackendRecord{{Kind: name, Name: name, Compiled: true, Available: true}}
}

// BackendIsAvailable reports whether kind is the backend in this build.
func BackendIsAvailable(kind string) bool { return kind == Backend() }

// DetectBestBackend reports the backend selected when the package was built.
func DetectBestBackend() string { return Backend() }
