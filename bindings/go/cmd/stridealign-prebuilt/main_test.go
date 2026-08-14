package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
)

func testArchive(t *testing.T, name string, contents string) []byte {
	t.Helper()
	var result bytes.Buffer
	compressed := gzip.NewWriter(&result)
	archive := tar.NewWriter(compressed)
	if err := archive.WriteHeader(&tar.Header{
		Name: name,
		Mode: 0o644,
		Size: int64(len(contents)),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := archive.Write([]byte(contents)); err != nil {
		t.Fatal(err)
	}
	if err := archive.Close(); err != nil {
		t.Fatal(err)
	}
	if err := compressed.Close(); err != nil {
		t.Fatal(err)
	}
	return result.Bytes()
}

func TestSafeExtractAcceptsOnePackageRoot(t *testing.T) {
	destination := t.TempDir()
	root, err := safeExtract(
		testArchive(t, "stride-align-go/lib/pkgconfig/stride-align-go.pc", "Version: 0.6.0\n"),
		destination,
	)
	if err != nil {
		t.Fatal(err)
	}
	contents, err := os.ReadFile(filepath.Join(root, "lib", "pkgconfig", "stride-align-go.pc"))
	if err != nil {
		t.Fatal(err)
	}
	if string(contents) != "Version: 0.6.0\n" {
		t.Fatalf("unexpected extracted contents %q", contents)
	}
}

func TestSafeExtractRejectsTraversal(t *testing.T) {
	_, err := safeExtract(testArchive(t, "../outside", "bad"), t.TempDir())
	if err == nil {
		t.Fatal("unsafe archive path was accepted")
	}
}

func TestSelectArtifactReportsAvailableProfiles(t *testing.T) {
	repository := manifest{Artifacts: []artifact{
		{Platform: "linux_amd64", Profile: "generic"},
		{Platform: "linux_amd64", Profile: "avx2"},
	}}
	selected, err := selectArtifact(repository, "linux_amd64", "avx2")
	if err != nil {
		t.Fatal(err)
	}
	if selected.Profile != "avx2" {
		t.Fatalf("selected profile %q", selected.Profile)
	}
	if _, err := selectArtifact(repository, "linux_amd64", "avx512bwvl"); err == nil {
		t.Fatal("missing profile did not produce an error")
	}
}
