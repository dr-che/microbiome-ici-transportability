Step 10.5D analysis completed

{
  "step": "Step10.5D",
  "version": "v1_V1_functional_sensitivity",
  "technical_status": "PASS",
  "primary_V2_pathway_gate": "HOLD_EXACT_UNSTRATIFIED_HUMANN3_MATRICES_UNRESOLVED",
  "analysis_role": "Separate V1 processing-system functional sensitivity; not concatenated with V2 species analysis",
  "v1_samples": 276,
  "v1_studies": 5,
  "community_unstratified_pathways_input": 319,
  "pathways_meta_analyzed": 319,
  "paired_species": 234,
  "pathway_meta": {
    "nominal_p_lt_0_05": 0,
    "nominal_p_lt_0_10": 1,
    "fdr_q_lt_0_10": 0,
    "minimum_fdr_q": 0.941640018265116,
    "I2_ge_50": 17,
    "strong_reversal": 3,
    "study_specific": 21
  },
  "layer_comparison": [
    {
      "layer": "Community pathways",
      "features_meta_analyzed": 319,
      "nominal_p_lt_0_05": 0,
      "nominal_p_lt_0_10": 1,
      "fdr_q_lt_0_10": 0,
      "median_I2": 4.8034124746456675,
      "I2_ge_50": 17,
      "I2_ge_75": 1,
      "mean_sign_consistency": 0.7172413793103447,
      "strong_reversal": 3,
      "study_specific": 21
    },
    {
      "layer": "Species",
      "features_meta_analyzed": 216,
      "nominal_p_lt_0_05": 2,
      "nominal_p_lt_0_10": 6,
      "fdr_q_lt_0_10": 0,
      "median_I2": 0.0,
      "I2_ge_50": 23,
      "I2_ge_75": 1,
      "mean_sign_consistency": 0.7183641975308643,
      "strong_reversal": 2,
      "study_specific": 25
    }
  ],
  "heterogeneity_distribution_test": {
    "mann_whitney_U": 37292.0,
    "p_value": 0.08218565662678702
  },
  "v2_candidate_mapping": {
    "total_candidates": 22,
    "mapped_exact_to_v1": 10
  },
  "species_pathway_concordance": {
    "all_pairs": 74646,
    "partial_spearman_fdr_lt_0_05": 10532,
    "mapped_candidate_strong_pairs": 44,
    "multi_candidate_convergent_pathways": 0
  },
  "scientific_decisions": {
    "functional_layer_more_stable_than_species": "TO_BE_INTERPRETED_FROM_RESULTS",
    "FDR_supported_universal_pathway": "NO",
    "pathway_direction_reversal": "SUPPORTED",
    "species_pathway_concordance": "SUPPORTED_FOR_SPECIFIC_PAIRS_NOT_UNIVERSAL",
    "V2_primary_species_pathway_integration": "NO_GO_UNTIL_EXACT_V2_PATHWAY_MATRICES",
    "V1_functional_sensitivity": "GO"
  },
  "limitations": [
    "V1 uses a separate MetaPhlAn3/HUMAnN3 processing system and five-study sample set.",
    "V1 response harmonization is not identical to the V2 363-patient endpoint lock.",
    "Lee sites are not separable in the V1 functional matrix.",
    "Metagenomic pathways represent functional potential, not transcription or metabolite production.",
    "Taxon-stratified rows were excluded from primary pathway meta-analysis."
  ]
}

TOP NOMINAL PATHWAYS