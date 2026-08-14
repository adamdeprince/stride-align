// Command stridealign-prebuilt installs a checksum-verified precompiled native
// backend for the stride-align Go package. The public Go API is still obtained
// from the normal source module; this command only replaces its C++23 build.
package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

const (
	projectVersion       = "0.6.0"
	repositoryURL        = "https://distribution.goblinreactor.com/stride-align/go/"
	maximumSize          = 250 << 20
	maximumExtractedSize = 1 << 30
)

type artifact struct {
	Platform   string `json:"platform"`
	Profile    string `json:"profile"`
	Path       string `json:"path"`
	SHA256     string `json:"sha256"`
	Validation string `json:"validation"`
}

type manifest struct {
	Project        string     `json:"project"`
	ProjectVersion string     `json:"project_version"`
	Artifacts      []artifact `json:"artifacts"`
}

func platformName() (string, error) {
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "darwin/arm64":
		return "darwin_arm64", nil
	case "linux/amd64":
		return "linux_amd64", nil
	case "linux/arm64":
		return "linux_arm64", nil
	case "linux/ppc64le":
		return "linux_ppc64le", nil
	case "linux/loong64":
		return "linux_loong64", nil
	default:
		return "", fmt.Errorf("no precompiled backend for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

func defaultProfile(platform string) string {
	switch platform {
	case "darwin_arm64", "linux_arm64":
		return "neon"
	case "linux_amd64":
		return "generic"
	case "linux_ppc64le":
		return "power8_vsx"
	case "linux_loong64":
		return "la464_lsx"
	default:
		return "generic"
	}
}

func profileTags(profile string) string {
	tags := []string{"stridealign_prebuilt"}
	switch profile {
	case "avx2":
		tags = append(tags, "stridealign_avx2")
	case "avx512bwvl":
		tags = append(tags, "stridealign_avx512bwvl")
	case "power8_vsx":
		tags = append(tags, "stridealign_power8_vsx")
	case "la464_lsx":
		tags = append(tags, "stridealign_la464_lsx")
	case "la464_lasx":
		tags = append(tags, "stridealign_la464_lasx")
	case "la664_lasx":
		tags = append(tags, "stridealign_la664_lasx")
	}
	return strings.Join(tags, ",")
}

func get(ctx context.Context, client *http.Client, url string) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: %s", url, response.Status)
	}
	contents, err := io.ReadAll(io.LimitReader(response.Body, maximumSize+1))
	if err != nil {
		return nil, err
	}
	if len(contents) > maximumSize {
		return nil, fmt.Errorf("download from %s exceeds %d bytes", url, maximumSize)
	}
	return contents, nil
}

func loadManifest(ctx context.Context, client *http.Client, baseURL string) (manifest, error) {
	contents, err := get(ctx, client, baseURL+"manifest.json")
	if err != nil {
		return manifest{}, err
	}
	var result manifest
	if err := json.Unmarshal(contents, &result); err != nil {
		return manifest{}, fmt.Errorf("decode repository manifest: %w", err)
	}
	if result.Project != "stride-align" || result.ProjectVersion != projectVersion {
		return manifest{}, fmt.Errorf(
			"repository serves %s %s, but this installer requires stride-align %s",
			result.Project, result.ProjectVersion, projectVersion,
		)
	}
	return result, nil
}

func selectArtifact(repository manifest, platform, profile string) (artifact, error) {
	available := make([]string, 0)
	for _, candidate := range repository.Artifacts {
		if candidate.Platform != platform {
			continue
		}
		available = append(available, candidate.Profile)
		if candidate.Profile == profile {
			return candidate, nil
		}
	}
	sort.Strings(available)
	if len(available) == 0 {
		return artifact{}, fmt.Errorf("no precompiled backends are published for %s", platform)
	}
	return artifact{}, fmt.Errorf(
		"profile %q is not published for %s; choose one of: %s",
		profile, platform, strings.Join(available, ", "),
	)
}

func verify(contents []byte, expected string) error {
	digest := sha256.Sum256(contents)
	actual := hex.EncodeToString(digest[:])
	if !strings.EqualFold(actual, expected) {
		return fmt.Errorf("SHA-256 mismatch: expected %s, downloaded %s", expected, actual)
	}
	return nil
}

func safeExtract(contents []byte, destination string) (string, error) {
	compressed, err := gzip.NewReader(bytes.NewReader(contents))
	if err != nil {
		return "", fmt.Errorf("open gzip archive: %w", err)
	}
	defer compressed.Close()
	archive := tar.NewReader(compressed)
	root := ""
	var extractedSize int64
	for {
		header, err := archive.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return "", fmt.Errorf("read archive: %w", err)
		}
		if header.Size < 0 || header.Size > maximumExtractedSize-extractedSize {
			return "", fmt.Errorf("archive expands beyond %d bytes", maximumExtractedSize)
		}
		extractedSize += header.Size
		name := filepath.Clean(filepath.FromSlash(header.Name))
		if name == "." || filepath.IsAbs(name) || name == ".." || strings.HasPrefix(name, ".."+string(filepath.Separator)) {
			return "", fmt.Errorf("archive contains unsafe path %q", header.Name)
		}
		parts := strings.Split(name, string(filepath.Separator))
		if root == "" {
			root = parts[0]
		} else if root != parts[0] {
			return "", fmt.Errorf("archive contains more than one top-level directory")
		}
		target := filepath.Join(destination, name)
		relative, err := filepath.Rel(destination, target)
		if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			return "", fmt.Errorf("archive path %q escapes installation directory", header.Name)
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return "", err
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return "", err
			}
			output, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
			if err != nil {
				return "", err
			}
			_, copyErr := io.Copy(output, archive)
			closeErr := output.Close()
			if copyErr != nil {
				return "", copyErr
			}
			if closeErr != nil {
				return "", closeErr
			}
		default:
			return "", fmt.Errorf("archive contains unsupported entry %q", header.Name)
		}
	}
	if root == "" {
		return "", fmt.Errorf("downloaded archive is empty")
	}
	return filepath.Join(destination, root), nil
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'"
}

func main() {
	profile := flag.String("profile", "", "CPU profile; the conservative platform default is used when omitted")
	installDir := flag.String("dir", "", "installation directory (default: ~/.stride-align/go)")
	list := flag.Bool("list", false, "list published profiles for this platform")
	flag.Parse()

	platform, err := platformName()
	if err != nil {
		fmt.Fprintln(os.Stderr, "stridealign-prebuilt:", err)
		os.Exit(1)
	}
	if *profile == "" {
		*profile = defaultProfile(platform)
	}
	if *installDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Fprintln(os.Stderr, "stridealign-prebuilt: find home directory:", err)
			os.Exit(1)
		}
		*installDir = filepath.Join(home, ".stride-align", "go")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	client := &http.Client{Timeout: 5 * time.Minute}
	repository, err := loadManifest(ctx, client, repositoryURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, "stridealign-prebuilt:", err)
		os.Exit(1)
	}
	if *list {
		profiles := make([]string, 0)
		for _, candidate := range repository.Artifacts {
			if candidate.Platform == platform {
				profiles = append(profiles, candidate.Profile)
			}
		}
		sort.Strings(profiles)
		fmt.Println(strings.Join(profiles, "\n"))
		return
	}

	selected, err := selectArtifact(repository, platform, *profile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "stridealign-prebuilt:", err)
		os.Exit(1)
	}
	contents, err := get(ctx, client, repositoryURL+selected.Path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "stridealign-prebuilt:", err)
		os.Exit(1)
	}
	if err := verify(contents, selected.SHA256); err != nil {
		fmt.Fprintln(os.Stderr, "stridealign-prebuilt:", err)
		os.Exit(1)
	}

	if err := os.MkdirAll(*installDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "stridealign-prebuilt:", err)
		os.Exit(1)
	}
	temporary, err := os.MkdirTemp(*installDir, ".install-")
	if err != nil {
		fmt.Fprintln(os.Stderr, "stridealign-prebuilt:", err)
		os.Exit(1)
	}
	defer os.RemoveAll(temporary)
	extracted, err := safeExtract(contents, temporary)
	if err != nil {
		fmt.Fprintln(os.Stderr, "stridealign-prebuilt:", err)
		os.Exit(1)
	}
	final := filepath.Join(*installDir, filepath.Base(extracted))
	if _, err := os.Stat(final); err == nil {
		fmt.Fprintf(os.Stderr, "stridealign-prebuilt: %s is already installed\n", final)
	} else if !errors.Is(err, os.ErrNotExist) {
		fmt.Fprintln(os.Stderr, "stridealign-prebuilt:", err)
		os.Exit(1)
	} else if err := os.Rename(extracted, final); err != nil {
		fmt.Fprintln(os.Stderr, "stridealign-prebuilt:", err)
		os.Exit(1)
	}

	pcdir := filepath.Join(final, "lib", "pkgconfig")
	fmt.Printf("Installed %s/%s in %s\n\n", platform, selected.Profile, final)
	fmt.Println("Build stride-align with:")
	fmt.Printf("export PKG_CONFIG_PATH=%s\"${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}\"\n", shellQuote(pcdir))
	fmt.Printf("go build -tags %s ./...\n", shellQuote(profileTags(selected.Profile)))
}
