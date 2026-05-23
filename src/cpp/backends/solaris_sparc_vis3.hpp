#pragma once

#include <sys/auxv.h>
#include <sys/auxv_SPARC.h>

#include <cstdint>

#include <nanobind/nanobind.h>

#include "backends/sparc_vis3_backend.hpp"
#include <vector>
#include <stride_align/levenshtein.hpp>

#include "cpu.hpp"
#include "cdist_simd.hpp"
#include "cdist_threshold.hpp"
#include "cdist_topk.hpp"
#include "jaro_simd.hpp"
#include "levenshtein_simd.hpp"
#include "levenshtein_simd_ops.hpp"
#include "osa_simd.hpp"

namespace stride_align::backend_solaris_sparc_vis3 {

namespace nb = nanobind;

using TargetImplementation = stride_align::sparc_vis3_backend::TargetImplementation;

struct Implementation {
  using PreparedSmithWatermanFarrarScore =
      TargetImplementation::PreparedSmithWatermanFarrarScore;
  using PreparedAffineScore = TargetImplementation::PreparedAffineScore;

  // Verify the runtime CPU exposes VIS3 (Oracle: vis3, vis3b, vis3c).
  // getisax(2) returns a bitmask of AV_SPARC_* feature flags.
  static bool supported_on_this_machine() noexcept {
    std::uint32_t caps[2] = {0, 0};
    if (getisax(caps, 2) == 0) {
      return false;
    }
    return (caps[0] & AV_SPARC_VIS3) != 0;
  }

  static void ensure_supported() {
    if (supported_on_this_machine()) {
      return;
    }
    PyErr_SetString(PyExc_RuntimeError, "Solaris SPARC VIS3 backend is not available on this machine");
    throw nb::python_error();
  }

  static Score smith_waterman_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_score(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentResult smith_waterman_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_path(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentPath smith_waterman_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_path_info(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static Score smith_waterman_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_score(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static PreparedSmithWatermanFarrarScore prepare_smith_waterman_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_farrar_score(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static Score smith_waterman_farrar_score_prepared(
      PreparedSmithWatermanFarrarScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_farrar_score_prepared(prepared);
  }

  static Score smith_waterman_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static AlignmentResult smith_waterman_affine_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_path(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static AlignmentPath smith_waterman_affine_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_path_info(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static std::string smith_waterman_affine_cigar(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_cigar(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static Score smith_waterman_affine_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_affine_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static Score smith_waterman_affine_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_score_prepared(prepared);
  }

  static PreparedAffineScore prepare_smith_waterman_affine_farrar_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_smith_waterman_affine_farrar_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static Score smith_waterman_affine_farrar_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::smith_waterman_affine_farrar_score_prepared(prepared);
  }

  static PreparedAffineScore prepare_needleman_wunsch_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::prepare_needleman_wunsch_affine_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static Score needleman_wunsch_affine_score_prepared(PreparedAffineScore& prepared) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score_prepared(prepared);
  }

  static Score needleman_wunsch_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_score(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentResult needleman_wunsch_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_path(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static AlignmentPath needleman_wunsch_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score, Score gap_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_path_info(
        query, target, match_score, mismatch_score, gap_score, width);
  }

  static Score needleman_wunsch_affine_score(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_score(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static AlignmentResult needleman_wunsch_affine_path(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_path(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static AlignmentPath needleman_wunsch_affine_path_info(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_path_info(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  static std::string needleman_wunsch_affine_cigar(
      nb::handle query, nb::handle target,
      Score match_score, Score mismatch_score,
      Score gap_open_score, Score gap_extend_score, unsigned int width) {
    ensure_supported();
    return TargetImplementation::needleman_wunsch_affine_cigar(
        query, target, match_score, mismatch_score, gap_open_score, gap_extend_score, width);
  }

  // ---- Scorer SIMD batch methods (bit-parallel via Vis3Ops, 1 lane) ----

  static std::vector<Score> levenshtein_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    ensure_supported();
    return ::stride_align::levenshtein_simd::levenshtein_scores_simd<
        ::stride_align::levenshtein_simd::Vis3Ops>(query, targets, cutoff);
  }

  static std::vector<double> levenshtein_normalized_scores(
      nb::handle query,
      nb::handle targets,
      std::size_t cutoff = ::stride_align::levenshtein::kNoCutoff) {
    ensure_supported();
    return ::stride_align::levenshtein_simd::levenshtein_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Vis3Ops>(query, targets, cutoff);
  }

  static std::vector<Score> damerau_levenshtein_scores(
      nb::handle query, nb::handle targets) {
    ensure_supported();
    return ::stride_align::osa_simd::osa_scores_simd<
        ::stride_align::levenshtein_simd::Vis3Ops>(query, targets);
  }

  static std::vector<double> damerau_levenshtein_normalized_scores(
      nb::handle query, nb::handle targets) {
    ensure_supported();
    return ::stride_align::osa_simd::osa_normalized_scores_simd<
        ::stride_align::levenshtein_simd::Vis3Ops>(query, targets);
  }

  static std::vector<double> jaro_similarities(
      nb::handle query, nb::handle targets) {
    ensure_supported();
    return ::stride_align::jaro_simd::jaro_similarities_simd<
        ::stride_align::levenshtein_simd::Vis3Ops>(query, targets);
  }

  static std::vector<double> jaro_winkler_similarities(
      nb::handle query,
      nb::handle targets,
      double prefix_weight,
      double prefix_threshold,
      std::size_t prefix_cap) {
    ensure_supported();
    return ::stride_align::jaro_simd::jaro_winkler_similarities_simd<
        ::stride_align::levenshtein_simd::Vis3Ops>(
        query, targets, prefix_weight, prefix_threshold, prefix_cap);
  }

  static nb::object cdist(
      nb::handle queries, nb::handle targets, int scorer,
      nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return ::stride_align::cdist_simd::cdist_impl<
        ::stride_align::levenshtein_simd::Vis3Ops>(
        queries, targets, scorer, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_above_threshold(
      nb::handle queries, nb::handle targets, int scorer,
      double threshold, nb::object tqdm_factory, std::size_t cpu_count,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return ::stride_align::cdist_threshold::cdist_threshold_impl<
        ::stride_align::levenshtein_simd::Vis3Ops>(
        queries, targets, scorer, threshold, tqdm_factory, cpu_count,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static nb::object cdist_top_k(
      nb::handle queries, nb::handle targets, int scorer,
      std::size_t k, nb::object tqdm_factory, std::size_t cpu_count,
      bool reject_duplicates,
      double jw_prefix_weight, double jw_prefix_threshold,
      std::size_t jw_prefix_cap) {
    ensure_supported();
    return ::stride_align::cdist_topk::cdist_top_k_impl<
        ::stride_align::levenshtein_simd::Vis3Ops>(
        queries, targets, scorer, k, tqdm_factory, cpu_count,
        reject_duplicates,
        jw_prefix_weight, jw_prefix_threshold, jw_prefix_cap);
  }

  static constexpr BackendKind backend_kind = BackendKind::solaris_sparc_vis3;
};

}  // namespace stride_align::backend_solaris_sparc_vis3
