import logging
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import pytest

WORKER = "PYTEST_XDIST_WORKER" in os.environ
LOG = logging.getLogger(__name__)
SUITE = {"heavy": "heavy-tests-" + os.environ.get("PYTEST_XDIST_WORKER", "serial")}
SETTINGS = dict(entities=40, steps=12, degree=2.0, max_degree=6, alive=0.8, lifetime=4.0, period=3)
SPLITS = dict(trained=["train"], scored=["validation"], sees=dict(train=[], validation=[]))
RECORD = Path(os.environ["GRAPHSPEC_SUITE_CHECKOUT"]) / "durations.json"
SPENT = {}
EARLY = []
ORIGINAL = {}


def suite_singleton(name, message):
    started = time.perf_counter()
    lock = ORIGINAL["singleton"](SUITE.get(name, name), message)
    LOG.debug("suite_singleton exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return lock


def suited():
    from graphspec.infra import compute
    from graphspec.infra.processes import locks

    started = time.perf_counter()
    ORIGINAL.update(singleton=locks.singleton, heavy_job=locks.heavy_job)
    locks.singleton, locks.heavy_job = suite_singleton, shared_heavy_job
    compute.THREADS = 1
    LOG.debug("suited exit seconds=%.4f", time.perf_counter() - started)


SLOW = {
    "tests/test_across_processes.py::test_an_entry_registered_in_every_registry_reaches_the_workers_of_a_build_and_a_fit",
    "tests/test_decisions.py::test_a_bank_of_decisions_completes_one_fit_reading_the_task_and_the_outcome_of_the_chosen_option",
    "tests/test_segments.py::test_the_arrays_of_a_joined_group_are_equal_in_one_process_and_on_one_and_two_workers",
    "tests/test_segments.py::test_random_pipelines_and_proposals_of_every_operation_derive_on_every_shared_bank_whole_and_in_segments",
    "tests/test_bayesian.py::test_acquisition_conditions_remaining_choices_on_epochs_and_runtime",
    "tests/test_decisions.py::test_every_decision_reads_back_as_stated_with_its_fields_in_the_type_of_their_column",
    "tests/test_edited.py::test_an_edit_completes_one_fit_whose_guards_record_that_messages_pass_back",
    "tests/test_pipelines.py::test_the_seven_in_todays_order_give_the_arrays_of_the_fixture_after_each_stage_and_the_representation",
    "tests/test_processes.py::test_windows_owner_death_terminates_its_job",
    "tests/test_bayesian.py::test_remaining_parameter_changes_with_the_fixed_parameter",
    "tests/test_segments.py::test_n_edits_at_one_time_take_one_derivation_of_the_history_and_n_of_the_last_step",
    "tests/test_scoring.py::test_k_options_take_one_encoding_of_the_history_and_k_of_the_last_step_and_one_encoding_per_option_does_not",
    "tests/test_time.py::test_a_case_of_time_reads_back_from_the_bank_equal_to_what_was_stated[forms]",
    "tests/test_library.py::test_database_export_and_staging_read_the_same_graph",
    "tests/test_edited.py::test_the_facts_an_edit_adds_come_from_the_content_of_its_option_alone",
    "tests/test_windows_mutex.py::test_singleton_excludes_other_home_and_releases",
    "tests/test_offered.py::test_every_value_a_search_offers_a_family_packs_a_batch_and_passes_forward_and_backward[snapshot]",
    "tests/test_forest.py::test_forest_suggest_records_ranked_candidates_and_fixed_parameters",
    "tests/test_edited.py::test_every_guard_holds_for_an_edit_and_one_refuses_a_field_whose_facts_the_family_does_not_read",
    "tests/test_hidden_layers.py::test_walks_accept_every_hidden_width_with_finite_gradients",
    "tests/test_references.py::test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name[snapshot]",
    "tests/test_standing.py::test_every_function_logs_its_time_or_is_compiled_or_is_listed_with_its_calls_per_run",
    "tests/test_time.py::test_a_database_gives_the_facts_of_two_clocks_and_their_observers_as_the_staging_does",
    "tests/test_draws.py::test_generated_options_are_drawn_by_keys_equal_when_entities_are_renumbered_and_rows_and_graphs_reordered",
    "tests/test_processes.py::test_windows_timeout_joins_worker_and_descendants",
    "tests/test_staged.py::test_sharded_staging_with_block_ids_builds_the_same_bank",
    "tests/test_segments.py::test_contraction_merge_and_difference_in_a_joined_group_equal_the_physical_join_of_its_graphs",
    "tests/test_forest.py::test_forest_prefers_the_low_loss_region_and_reports_disagreement",
    "tests/test_draws.py::test_permuting_entities_and_graphs_leaves_every_derived_array_equal_after_sorting_by_key",
    "tests/test_processes.py::test_windows_failed_job_attachment_exits_before_target_starts",
    "tests/test_offered.py::test_every_value_a_search_offers_a_family_packs_a_batch_and_passes_forward_and_backward[memory]",
    "tests/test_offered.py::test_every_value_a_search_offers_a_family_packs_a_batch_and_passes_forward_and_backward[event_graph]",
    "tests/test_library.py::test_heavy_jobs_are_exclusive_across_processes",
    "tests/test_packing.py::test_packer_matches_single_process[event_graph]",
    "tests/test_realisations.py::test_nothing_a_realisation_must_not_read_reaches_its_batch",
    "tests/test_model_config.py::test_the_check_states_every_value_of_a_choice_without_a_default_and_starts_its_switches_from_each",
    "tests/test_storage.py::test_child_initializes_storage_before_creating_temporary_files",
    "tests/test_selector.py::test_random_epoch_ranges_persist_and_use_inclusive_integer_draws",
    "tests/test_segments.py::test_every_operation_derives_steps_that_hold_nothing_in_segments_as_it_derives_them_whole",
    "tests/test_windows_mutex.py::test_abandoned_mutex_is_owned_and_released",
    "tests/test_segments.py::test_a_fork_its_sibling_forks_and_the_trunk_equal_the_whole_graphs_derived_separately",
    "tests/test_realisations.py::test_each_guard_refuses_its_broken_realisation_before_a_fit_naming_the_field_the_kind_of_task_and_it",
    "tests/test_edited.py::test_the_chains_and_forks_a_batch_reads_are_released_once_it_is_packed",
    "tests/test_decisions.py::test_no_option_without_an_outcome_and_no_option_not_chosen_reaches_a_loss_as_a_target",
    "tests/test_decisions.py::test_a_node_or_an_edge_an_action_targets_unobserved_at_the_step_of_its_decision_stops_the_build",
    "tests/test_decisions.py::test_a_target_or_a_generator_that_does_not_resolve_stops_the_build_naming_them",
    "tests/test_leakage.py::test_the_scored_tasks_a_trained_split_hides_shape_neither_its_steps_nor_its_predictions",
    "tests/test_forest.py::test_log_expected_improvement_matches_the_closed_form_and_stays_finite_far_away",
    "tests/test_bayesian.py::test_selector_mix_rolls_each_trial_after_startup",
    "tests/test_small.py::test_a_view_without_any_edge_completes_one_fit_on_cpu[bare]",
    "tests/test_offered.py::test_every_value_a_search_offers_a_family_packs_a_batch_and_passes_forward_and_backward[sequence]",
    "tests/test_edited.py::test_an_option_whose_edit_the_variant_shows_none_of_at_its_step_is_left_out_and_counted",
    "tests/test_offered.py::test_every_value_a_search_offers_a_family_packs_a_batch_and_passes_forward_and_backward[walks]",
    "tests/test_offered.py::test_every_family_reads_the_kinds_relations_and_columns_a_pipeline_adds[snapshot]",
    "tests/test_leakage.py::test_encoders_are_fitted_on_what_the_views_of_the_splits_trained_on_show_and_on_no_other",
    "tests/test_bayesian.py::test_history_and_candidates_preserve_parameters_and_condition_on_random_values",
    "tests/test_nodes.py::test_every_kind_naming_nodes_reads_back_as_stated_and_its_points_name_their_nodes",
    "tests/test_edited.py::test_each_option_scores_as_its_own_graph_encoded_alone_and_one_step_trains_every_family_that_reads_an_edit[snapshot]",
    "tests/test_tasks.py::test_a_view_holds_times_without_a_task_and_several_tasks_at_one_time",
    "tests/test_schema.py::test_the_varied_graph_completes_one_fit_on_cpu",
    "tests/test_offered.py::test_every_value_a_search_offers_a_family_packs_a_batch_and_passes_forward_and_backward[projection]",
    "tests/test_selector.py::test_a_search_offers_no_pipeline_its_statement_refuses_and_bounds_that_leave_nothing_stop_naming_the_entry",
    "tests/test_pipelines.py::test_a_pipeline_in_another_order_or_with_an_operation_twice_completes_one_fit[twice]",
    "tests/test_offered.py::test_every_family_reads_the_kinds_relations_and_columns_a_pipeline_adds[memory]",
    "tests/test_hidden_layers.py::test_attention_accepts_every_hidden_width_with_finite_gradients[4]",
    "tests/test_realisations.py::test_each_shipped_realisation_passes_every_guard_on_its_shared_bank_and_completes_one_fit[set_edge]",
    "tests/test_pipelines.py::test_a_pipeline_in_another_order_or_with_an_operation_twice_completes_one_fit[reordered]",
    "tests/test_packing.py::test_packer_matches_single_process[snapshot]",
    "tests/test_offered.py::test_every_value_a_search_offers_a_family_packs_a_batch_and_passes_forward_and_backward[tree]",
    "tests/test_hidden_layers.py::test_attention_accepts_every_hidden_width_with_finite_gradients[3]",
    "tests/test_edited.py::test_each_option_scores_as_its_own_graph_encoded_alone_and_one_step_trains_every_family_that_reads_an_edit[tree]",
    "tests/test_selector.py::test_tpe_learns_from_every_recorded_trial_inside_the_bounds",
    "tests/test_operations.py::test_each_operation_called_alone_on_the_fixture_gives_the_arrays_of_its_stage[7]",
    "tests/test_offered.py::test_every_value_a_search_offers_a_family_packs_a_batch_and_passes_forward_and_backward[mixed]",
    "tests/test_added.py::test_a_representation_registered_in_each_family_completes_one_fit_on_cpu",
    "tests/test_time.py::test_a_case_of_time_completes_one_fit_on_cpu[seen]",
    "tests/test_tasks.py::test_a_bank_of_four_splits_with_targets_on_nodes_and_on_graphs_completes_one_fit",
    "tests/test_realisations.py::test_each_shipped_realisation_passes_every_guard_on_its_shared_bank_and_completes_one_fit[actor_edge]",
    "tests/test_offered.py::test_every_family_reads_the_kinds_relations_and_columns_a_pipeline_adds[event_graph]",
    "tests/test_forest.py::test_failed_trials_train_the_completion_forest",
    "tests/test_composition.py::test_a_prediction_depends_neither_on_the_windows_beside_it_nor_on_a_fact_after_its_time[snapshot]",
    "tests/test_time.py::test_a_case_of_time_completes_one_fit_on_cpu[forms]",
    "tests/test_sets.py::test_each_form_of_storage_completes_one_fit_on_cpu[pairs]",
    "tests/test_sets.py::test_each_form_of_storage_completes_one_fit_on_cpu[node]",
    "tests/test_offered.py::test_every_family_reads_the_kinds_relations_and_columns_a_pipeline_adds[walks]",
    "tests/test_offered.py::test_every_family_reads_the_kinds_relations_and_columns_a_pipeline_adds[mixed]",
    "tests/test_hidden_layers.py::test_attention_accepts_every_hidden_width_with_finite_gradients[2]",
    "tests/test_decisions.py::test_every_family_trains_on_the_outcome_of_the_chosen_option[mixed]",
    "tests/test_classes.py::test_a_bank_of_a_number_and_kinds_over_classes_completes_one_fit",
    "tests/test_time.py::test_facts_about_one_entity_at_one_time_stop_naming_it_where_no_rule_orders_them",
    "tests/test_time.py::test_a_form_is_equal_to_the_same_content_stated_as_snapshots[0]",
    "tests/test_targets.py::test_a_bank_of_numbers_and_paths_completes_one_fit_with_a_head_per_kind_of_target",
    "tests/test_realisations.py::test_each_shipped_realisation_passes_every_guard_on_its_shared_bank_and_completes_one_fit[vector]",
    "tests/test_realisations.py::test_each_shipped_realisation_passes_every_guard_on_its_shared_bank_and_completes_one_fit[action_nodes]",
    "tests/test_outcomes.py::test_every_kind_of_2_13_reads_back_as_stated",
    "tests/test_decisions.py::test_availability_and_what_happened_never_reach_a_models_input",
    "tests/test_targets.py::test_the_batch_a_model_reads_holds_the_inputs_of_a_path_and_of_a_registered_kind_and_no_outcome",
    "tests/test_selector.py::test_new_trial_limits_are_frozen_and_epochs_can_reach_five",
    "tests/test_segments.py::test_a_joined_group_derived_in_segments_at_any_times_equals_its_physical_join[0]",
    "tests/test_realisations.py::test_each_option_scores_as_it_does_packed_alone_and_one_step_trains_every_family_under_each_shipped_realisation[mixed]",
    "tests/test_edited.py::test_a_batch_of_forks_carries_what_its_reader_needs_so_only_its_packer_derives_them",
    "tests/test_selector.py::test_running_again_continues_and_may_change_selector_random_phase_and_bounds",
    "tests/test_selector.py::test_random_and_tpe_accept_non_power_of_two_settings",
    "tests/test_realisations.py::test_each_option_scores_as_it_does_packed_alone_and_one_step_trains_every_family_under_each_shipped_realisation[tree]",
    "tests/test_outcomes.py::test_a_bank_of_the_kinds_of_2_13_completes_one_fit",
    "tests/test_operations.py::test_each_operation_called_alone_on_the_fixture_gives_the_arrays_of_its_stage[4]",
    "tests/test_library.py::test_public_graph_writer_builds_without_database_and_preserves_source",
    "tests/test_targets.py::test_a_kind_of_target_registered_by_a_test_builds_reads_back_and_completes_one_fit",
    "tests/test_standing.py::test_only_the_registries_read_what_a_registry_holds_and_other_code_reads_it_by_its_lookup",
    "tests/test_references.py::test_a_relation_taken_from_a_column_has_the_parameters_arrays_and_weights_of_its_table",
    "tests/test_realisations.py::test_each_option_scores_as_it_does_packed_alone_and_one_step_trains_every_family_under_each_shipped_realisation[snapshot]",
    "tests/test_realisations.py::test_each_option_scores_as_it_does_packed_alone_and_one_step_trains_every_family_under_each_shipped_realisation[event_graph]",
    "tests/test_path.py::test_path_windows_join_like_the_windows_of_numbers",
    "tests/test_operations.py::test_each_operation_called_alone_on_the_fixture_gives_the_arrays_of_its_stage[5]",
    "tests/test_nodes.py::test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged[memory]",
    "tests/test_nodes.py::test_a_link_whose_nodes_the_pipeline_does_not_keep_is_left_out_and_its_kept_candidates_passed_alike",
    "tests/test_decisions.py::test_every_family_trains_on_the_outcome_of_the_chosen_option[walks]",
    "tests/test_bayesian.py::test_failed_trials_train_feasibility_without_fake_objective_values",
    "tests/test_references.py::test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name[mixed]",
    "tests/test_references.py::test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name[event_graph]",
    "tests/test_references.py::test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name[memory]",
    "tests/test_references.py::test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name[projection]",
    "tests/test_references.py::test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name[sequence]",
    "tests/test_references.py::test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name[tree]",
    "tests/test_references.py::test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name[walks]",
    "tests/test_references.py::test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name[affinity]",
    "tests/test_references.py::test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name[signal]",
    "tests/test_references.py::test_reordering_the_graphs_of_a_bank_leaves_the_predictions_of_the_projection_family_equal",
    "tests/test_references.py::test_renumbering_entities_and_reordering_graphs_leaves_the_batches_and_predictions_of_every_family_equal[snapshot]",
    "tests/test_references.py::test_renumbering_entities_and_reordering_graphs_leaves_the_batches_and_predictions_of_every_family_equal[mixed]",
    "tests/test_references.py::test_renumbering_entities_and_reordering_graphs_leaves_the_batches_and_predictions_of_every_family_equal[event_graph]",
    "tests/test_references.py::test_renumbering_entities_and_reordering_graphs_leaves_the_batches_and_predictions_of_every_family_equal[memory]",
    "tests/test_references.py::test_renumbering_entities_and_reordering_graphs_leaves_the_batches_and_predictions_of_every_family_equal[projection]",
    "tests/test_references.py::test_renumbering_entities_and_reordering_graphs_leaves_the_batches_and_predictions_of_every_family_equal[sequence]",
    "tests/test_references.py::test_renumbering_entities_and_reordering_graphs_leaves_the_batches_and_predictions_of_every_family_equal[tree]",
    "tests/test_references.py::test_renumbering_entities_and_reordering_graphs_leaves_the_batches_and_predictions_of_every_family_equal[walks]",
    "tests/test_references.py::test_renumbering_entities_and_reordering_graphs_leaves_the_batches_and_predictions_of_every_family_equal[affinity]",
    "tests/test_references.py::test_renumbering_entities_and_reordering_graphs_leaves_the_batches_and_predictions_of_every_family_equal[signal]",
    "tests/test_references.py::test_the_kind_of_a_task_reaches_every_family[snapshot]",
    "tests/test_references.py::test_the_kind_of_a_task_reaches_every_family[mixed]",
    "tests/test_references.py::test_the_kind_of_a_task_reaches_every_family[event_graph]",
    "tests/test_references.py::test_the_kind_of_a_task_reaches_every_family[memory]",
    "tests/test_references.py::test_the_kind_of_a_task_reaches_every_family[projection]",
    "tests/test_references.py::test_the_kind_of_a_task_reaches_every_family[sequence]",
    "tests/test_references.py::test_the_kind_of_a_task_reaches_every_family[tree]",
    "tests/test_references.py::test_the_kind_of_a_task_reaches_every_family[walks]",
    "tests/test_references.py::test_the_kind_of_a_task_reaches_every_family[affinity]",
    "tests/test_references.py::test_the_kind_of_a_task_reaches_every_family[signal]",
    "tests/test_references.py::test_the_value_of_a_relation_taken_from_a_column_reaches_every_family_that_reads_edges[snapshot]",
    "tests/test_references.py::test_the_value_of_a_relation_taken_from_a_column_reaches_every_family_that_reads_edges[event_graph]",
    "tests/test_references.py::test_the_value_of_a_relation_taken_from_a_column_reaches_every_family_that_reads_edges[memory]",
    "tests/test_references.py::test_the_value_of_a_relation_taken_from_a_column_reaches_every_family_that_reads_edges[projection]",
    "tests/test_references.py::test_the_value_of_a_relation_taken_from_a_column_reaches_every_family_that_reads_edges[sequence]",
    "tests/test_references.py::test_the_value_of_a_relation_taken_from_a_column_reaches_every_family_that_reads_edges[tree]",
    "tests/test_references.py::test_the_value_of_a_relation_taken_from_a_column_reaches_every_family_that_reads_edges[walks]",
    "tests/test_references.py::test_the_kinds_of_event_and_the_relations_never_share_a_slot",
    "tests/test_references.py::test_a_frame_of_a_sequence_over_graphs_reads_every_task_seen_as_an_event_at_the_step_before_it",
    "tests/test_references.py::test_the_attributes_of_a_graph_reach_every_family[snapshot]",
    "tests/test_references.py::test_the_attributes_of_a_graph_reach_every_family[mixed]",
    "tests/test_references.py::test_the_attributes_of_a_graph_reach_every_family[event_graph]",
    "tests/test_references.py::test_the_attributes_of_a_graph_reach_every_family[memory]",
    "tests/test_references.py::test_the_attributes_of_a_graph_reach_every_family[projection]",
    "tests/test_references.py::test_the_attributes_of_a_graph_reach_every_family[sequence]",
    "tests/test_references.py::test_the_attributes_of_a_graph_reach_every_family[tree]",
    "tests/test_references.py::test_the_attributes_of_a_graph_reach_every_family[walks]",
    "tests/test_references.py::test_the_attributes_of_a_graph_reach_every_family[affinity]",
    "tests/test_references.py::test_the_attributes_of_a_graph_reach_every_family[signal]",
    "tests/test_classes.py::test_a_class_is_one_of_at_least_two_stated_classes_and_holds_one_point_for_each",
    "tests/test_classes.py::test_a_class_is_predicted_by_scores_its_loss_normalises_per_task_and_its_measures_read_probabilities",
    "tests/test_classes.py::test_a_memory_reads_a_window_alone_that_holds_no_event",
    "tests/test_classes.py::test_a_proportion_vector_states_a_share_of_every_class_and_is_predicted_and_measured_as_one",
    "tests/test_classes.py::test_a_set_of_classes_is_one_decision_per_class_by_its_loss_and_its_measure",
    "tests/test_classes.py::test_every_kind_over_classes_matures_into_its_own_form_once_known",
    "tests/test_classes.py::test_every_kind_over_classes_reads_back_as_stated_with_its_classes_as_its_series",
    "tests/test_classes.py::test_the_batch_a_model_reads_holds_no_outcome_of_a_number_or_a_kind_over_classes",
    "tests/test_classes.py::test_what_matures_of_a_kind_over_classes_is_a_column_or_a_column_per_class_of_its_own_type",
    "tests/test_nodes.py::test_a_bank_of_kinds_naming_nodes_completes_one_fit",
    "tests/test_nodes.py::test_a_fit_whose_pipeline_draws_no_negatives_for_a_link_stops_naming_its_kind",
    "tests/test_nodes.py::test_a_link_from_or_to_a_node_not_observed_at_its_task_is_refused_when_the_bank_is_built",
    "tests/test_nodes.py::test_a_link_names_one_node_its_relation_joins_to_the_node_it_starts_from",
    "tests/test_nodes.py::test_a_link_withholds_its_target_edge_from_its_split_and_is_scored_among_the_negatives_of_its_task",
    "tests/test_nodes.py::test_a_link_withholds_its_target_edge_only_until_the_target_is_known",
    "tests/test_nodes.py::test_a_pipeline_that_removes_a_kind_a_link_names_is_refused_when_stated_and_never_offered",
    "tests/test_nodes.py::test_a_ranking_is_scored_listwise_and_measured_by_gain_and_reciprocal_rank",
    "tests/test_nodes.py::test_a_ranking_names_every_listed_item_once_and_no_other_node",
    "tests/test_nodes.py::test_a_ranking_states_a_relevance_for_every_item_its_task_lists",
    "tests/test_nodes.py::test_a_search_holds_the_negatives_a_link_needs_and_a_template_without_them_stops_naming_its_kind",
    "tests/test_nodes.py::test_a_set_of_nodes_is_one_decision_per_node_its_task_lists",
    "tests/test_nodes.py::test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged[affinity]",
    "tests/test_nodes.py::test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged[event_graph]",
    "tests/test_nodes.py::test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged[mixed]",
    "tests/test_nodes.py::test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged[projection]",
    "tests/test_nodes.py::test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged[sequence]",
    "tests/test_nodes.py::test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged[signal]",
    "tests/test_nodes.py::test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged[snapshot]",
    "tests/test_nodes.py::test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged[tree]",
    "tests/test_nodes.py::test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged[walks]",
    "tests/test_nodes.py::test_the_batch_a_model_reads_holds_no_outcome_of_a_kind_naming_nodes",
    "tests/test_nodes.py::test_the_draw_of_a_links_negatives_leaves_out_its_positive_under_every_shipped_rule",
    "tests/test_outcomes.py::test_a_kind_of_target_whose_values_are_times_is_refused_in_a_schema_of_several_units",
    "tests/test_outcomes.py::test_a_sequence_is_predicted_place_by_place_against_its_end_and_scored_per_token",
    "tests/test_outcomes.py::test_a_sequence_states_its_tokens_and_its_greatest_length_and_holds_a_point_per_token_and_place",
    "tests/test_outcomes.py::test_a_time_to_an_event_is_a_likelihood_over_intervals_that_reads_cases_ended_before_their_event",
    "tests/test_outcomes.py::test_a_time_to_an_event_states_its_duration_and_outcome_and_holds_a_point_per_interval",
    "tests/test_outcomes.py::test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged[affinity]",
    "tests/test_outcomes.py::test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged[event_graph]",
    "tests/test_outcomes.py::test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged[memory]",
    "tests/test_outcomes.py::test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged[mixed]",
    "tests/test_outcomes.py::test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged[projection]",
    "tests/test_outcomes.py::test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged[sequence]",
    "tests/test_outcomes.py::test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged[signal]",
    "tests/test_outcomes.py::test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged[snapshot]",
    "tests/test_outcomes.py::test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged[tree]",
    "tests/test_outcomes.py::test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged[walks]",
    "tests/test_outcomes.py::test_concordance_without_a_comparable_pair_among_the_tasks_scored_stops_a_fit_when_it_starts",
    "tests/test_outcomes.py::test_the_batch_a_model_reads_holds_the_inputs_of_a_sequence_and_a_time_to_an_event_and_no_outcome",
    "tests/test_outcomes.py::test_times_past_two_to_the_24_are_held_and_read_by_the_loss_and_the_measures_exactly",
    "tests/test_reach_fit.py::test_reach_names_the_options_and_fields_the_batches_of_a_fit_hold_of_the_stated_bank_whole_and_with_spots_sampled",
    "tests/test_reach_fit.py::test_reach_names_the_options_and_fields_the_batches_of_a_fit_hold_of_the_picked_bank_under_the_node_realisation",
    "tests/test_reach_fit.py::test_under_the_edit_and_the_setting_of_28_september_reach_names_the_facts_the_forks_of_a_fit_hold_and_the_value_that_gives_a_field_back",
    "tests/test_reach_fit.py::test_the_fields_the_guards_of_the_smallest_fit_see_are_those_reach_puts_before_the_model_for_an_option_held_in_the_scored_splits",
    "tests/test_reach_fit.py::test_a_bank_without_time_and_one_of_order_built_as_such_are_described_in_the_words_of_their_model_as_the_records_written_out_here",
}


def pytest_configure(config):
    started = time.perf_counter()
    config.addinivalue_line("markers", "gpu: the test is about the GPU and is skipped where none is present")
    config.addinivalue_line("markers", "slow: the test is left out of the default run; run it with -m slow")
    config.option.loadscopereorder = False
    LOG.debug("configure exit seconds=%.4f", time.perf_counter() - started)


def unit(nodeid):
    started = time.perf_counter()
    found = nodeid.rsplit("@", 1)[1] if nodeid.rfind("@") > nodeid.rfind("]") else nodeid
    LOG.debug("unit exit unit=%s seconds=%.6f", found, time.perf_counter() - started)
    return found


def ordered(items):
    from graphspec.infra.files import read_json

    started = time.perf_counter()
    spent = read_json(RECORD) if RECORD.exists() else {}
    units = [unit(item.nodeid) for item in items]
    counts, first = Counter(units), {}
    for position, name in enumerate(units):
        first.setdefault(name, position)
    rank = {name: (-spent.get(name, 0.0), -counts[name], first[name]) for name in counts}
    items[:] = [item for _, item in sorted(zip(units, items), key=lambda pair: rank[pair[0]])]
    LOG.debug(
        "ordered exit units=%d recorded=%d seconds=%.4f", len(counts), len(spent), time.perf_counter() - started
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items):
    import torch

    started = time.perf_counter()
    absent = pytest.mark.skip(reason="no GPU is present")
    marked = [item for item in items if "gpu" in item.keywords]
    if not torch.cuda.is_available():
        for item in marked:
            item.add_marker(absent)
    if WORKER:
        ordered(items)
    LOG.debug("gpu marks exit marked=%d seconds=%.4f", len(marked), time.perf_counter() - started)


def pytest_runtest_logreport(report):
    started = time.perf_counter()
    name = unit(report.nodeid)
    SPENT[name] = SPENT.get(name, 0.0) + report.duration
    LOG.debug("spent exit unit=%s seconds=%.6f", name, time.perf_counter() - started)


def made_early():
    started = time.perf_counter()
    shared("bank", lambda root: synthetic_bank(root, "task"))
    shared("path", lambda root: synthetic_bank(root, "path"))
    ground = Path(os.environ["GRAPHSPEC_SUITE_HOME"]) / "ground" / "fixture"
    shared("ground", lambda made: ground_banks(made, ground))
    shared("varied", varied_bank)
    LOG.debug("made early exit seconds=%.4f", time.perf_counter() - started)


@pytest.hookimpl(wrapper=True)
def pytest_sessionstart(session):
    result = yield
    started = time.perf_counter()
    suited()
    if not WORKER and session.config.pluginmanager.has_plugin("dsession"):
        EARLY.append(threading.Thread(target=made_early, daemon=True))
        EARLY[0].start()
    LOG.debug("session start exit early=%d seconds=%.4f", len(EARLY), time.perf_counter() - started)
    return result


def pytest_sessionfinish(session):
    from graphspec.infra.files import read_json, write_json

    started = time.perf_counter()
    if not WORKER and session.config.pluginmanager.has_plugin("dsession"):
        held = read_json(RECORD) if RECORD.exists() else {}
        write_json(RECORD, {**held, **{name: round(seconds, 4) for name, seconds in SPENT.items()}})
    LOG.debug("durations recorded exit units=%d seconds=%.4f", len(SPENT), time.perf_counter() - started)


@contextmanager
def shared_heavy_job():
    from graphspec.infra.processes.locks import LockBusy

    while True:
        job = ORIGINAL["heavy_job"]()
        try:
            job.__enter__()
            break
        except RuntimeError as error:
            if not isinstance(error.__cause__, LockBusy):
                raise
            time.sleep(0.05)
    try:
        yield
    finally:
        job.__exit__(None, None, None)


def suite():
    return Path(os.environ["GRAPHSPEC_SUITE_HOME"]) / "tmp" / "suite" / os.environ["GRAPHSPEC_SUITE_RUN"]


def shared(name, make):
    from graphspec.infra.processes.locks import Lock, LockBusy

    started = time.perf_counter()
    root = suite() / name
    root.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            with Lock(root.parent / (name + ".lock")):
                if not (root / "made").exists():
                    root.mkdir()
                    make(root)
                    (root / "made").write_text("")
            break
        except LockBusy:
            time.sleep(0.01)
    LOG.debug("shared exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return root


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    started = time.perf_counter()
    if WORKER:
        LOG.debug("worker ends without finalizing seconds=%.4f", time.perf_counter() - started)
        sys.stdout.flush()
        sys.stderr.flush()
        if os.name == "nt":
            import _winapi

            _winapi.TerminateProcess(_winapi.GetCurrentProcess(), 0)
        os._exit(0)
    for thread in EARLY:
        thread.join()
    LOG.debug("early threads joined seconds=%.4f", time.perf_counter() - started)


def synthetic_bank(root, level):
    from graphspec.sources.synthetic import generate, uniform
    from tuning.bank.build import build

    started = time.perf_counter()
    task = generate(root, "tiny", 4, 1, uniform(4, 2, level), **SETTINGS)
    build(task, root / "bank", 1, ("train",))
    LOG.debug("synthetic bank exit level=%s seconds=%.4f", level, time.perf_counter() - started)


@pytest.fixture(scope="session")
def bank():
    return shared("bank", lambda root: synthetic_bank(root, "task")) / "bank"


@pytest.fixture(scope="session")
def path_bank():
    return shared("path", lambda root: synthetic_bank(root, "path")) / "bank"


def varied_bank(root):
    from graphspec.sources import synthetic
    from tuning.bank.build import build

    started = time.perf_counter()
    task = synthetic.generate(root, "varied", 4, 1, synthetic.varied("task"), **SETTINGS)
    build(task, root / "bank", 1, ("train",))
    LOG.debug("varied bank exit seconds=%.4f", time.perf_counter() - started)


@pytest.fixture(scope="session")
def varied():
    return shared("varied", varied_bank) / "bank"


def sets_built(root):
    from test_sets import case

    from graphspec.sources import synthetic
    from tuning.bank.build import build

    started = time.perf_counter()
    build(synthetic.stage(root, "sets", case()), root / "bank", 1, ("train",))
    LOG.debug("sets built exit seconds=%.4f", time.perf_counter() - started)


@pytest.fixture(scope="session")
def sets_bank():
    started = time.perf_counter()
    result = shared("sets", sets_built) / "bank"
    LOG.debug("sets bank exit seconds=%.4f", time.perf_counter() - started)
    return result


@pytest.fixture(scope="session")
def ground():
    place = Path(os.environ["GRAPHSPEC_SUITE_HOME"]) / "ground" / "fixture"
    if not (place / "fixture.json").exists():
        raise FileNotFoundError("the fixture of the ground is absent: %s" % place)
    return place


def ground_banks(root, place):
    import fixture

    from graphspec.sources import synthetic
    from tuning.bank.build import build

    started = time.perf_counter()
    for entry in fixture.configurations(place)[0]:
        source = entry["source"]
        held = root / source["name"]
        if not held.exists():
            held.mkdir()
            shape, settings = fixture.shaped(source)
            task = synthetic.generate(held, "source", source["graphs"], source["seed"], shape, **settings)
            build(task, held / "bank", 1, ("train",))
    LOG.debug("ground banks exit seconds=%.4f", time.perf_counter() - started)


@pytest.fixture(scope="session")
def grounded(ground):
    import fixture

    root = shared("ground", lambda made: ground_banks(made, ground))
    return {entry["source"]["name"]: root / entry["source"]["name"] for entry in fixture.configurations(ground)[0]}


def stated(directory, view):
    from support import configuration, several

    from tuning.bank.bank import Bank

    started = time.perf_counter()
    registry = Bank(directory).registry
    cfg = configuration(registry, dict(view=view), {"time/history": 4, **several(registry)}, 5, SPLITS)
    LOG.debug("stated exit view=%s seconds=%.4f", view, time.perf_counter() - started)
    return cfg


def running(directory, view):
    from support import SEED

    from graphspec.infra.arrays import RESIDENT
    from tuning.bank.bank import Deriver
    from tuning.bank.variants import prepare
    from tuning.batches.packing import Packer

    started = time.perf_counter()
    cfg = stated(directory, view)
    packer = Packer(Deriver(directory, "cpu", SEED, RESIDENT), cfg, "cpu", 1)
    prepare(directory, cfg, 1, "cpu", SEED)
    LOG.debug("running exit view=%s seconds=%.4f", view, time.perf_counter() - started)
    return cfg, packer


def closed(held):
    started = time.perf_counter()
    for cfg, packer in held:
        packer.close()
    LOG.debug("closed exit packers=%d seconds=%.4f", len(held), time.perf_counter() - started)


@pytest.fixture(scope="session", autouse=True)
def warmup(request):
    if WORKER:
        return
    for name in ("packers", "candidate_workers"):
        if any(name in item.fixturenames for item in request.session.items):
            request.getfixturevalue(name)


@pytest.fixture(autouse=True)
def processes_early(request):
    started = time.perf_counter()
    grouped = request.node.get_closest_marker("xdist_group")
    early = dict(workers="packers", candidates="candidate_workers")
    if WORKER and grouped is not None and grouped.args[0] in early:
        request.getfixturevalue(early[grouped.args[0]])
    LOG.debug("processes early exit seconds=%.4f", time.perf_counter() - started)


@pytest.fixture(scope="session")
def packers(bank):
    held = {view: running(bank, view) for view in ("snapshot", "event_graph")}
    yield held
    closed(held.values())


@pytest.fixture
def statement():
    from support import SEED

    from tuning.statement import VERSION

    return dict(
        version=VERSION,
        device="cpu",
        seed=5,
        derivation_seed=SEED,
        seconds=30,
        preparation_seconds=300,
        workers=1,
        variant_cache_bytes=1 << 30,
        batch_sequences=16,
        **SPLITS,
        loss=dict(number="squared"),
        measures=["mse", "r2"],
        alike=1.0,
    )


class Inline:
    def __init__(self, bank, cfg, device, workers, timing_directory=None):
        if device != "cpu" or workers != 1:
            raise ValueError("the suite packs in its own process, on cpu, as one worker")
        self.bank, self.cfg = bank, cfg

    def stream(self, groups):
        from tuning.batches.batching import recipe_batch
        from tuning.batches.tensors import materialise

        started = time.perf_counter()
        for group in groups:
            yield materialise(self.bank, self.cfg, recipe_batch(self.bank, group, self.cfg), "cpu")
        LOG.debug("inline stream exit batches=%d seconds=%.4f", len(groups), time.perf_counter() - started)

    def close(self):
        pass


def inline(directory, views, monkeypatch):
    import torch

    started = time.perf_counter()
    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    monkeypatch.setattr("tuning.batches.packing.Packer", Inline)
    result = {view: dict(stated(directory, view), epochs=1) for view in views}
    LOG.debug("inline exit views=%d seconds=%.4f", len(views), time.perf_counter() - started)
    return result


@pytest.fixture
def packed(bank, monkeypatch):
    return inline(bank, ("snapshot", "event_graph"), monkeypatch)


@pytest.fixture
def varied_packed(varied, monkeypatch):
    return inline(varied, ("snapshot",), monkeypatch)


@pytest.fixture(scope="session")
def candidate_workers():
    started = time.perf_counter()
    executor = ProcessPoolExecutor(2)
    for _ in range(2):
        executor.submit(import_module, "tuning.searches.space")
    LOG.debug("candidate workers started seconds=%.4f", time.perf_counter() - started)
    yield executor
    executor.shutdown()


@pytest.fixture
def candidate_pool(candidate_workers):
    from graphspec import registries
    from tuning.searches import space

    started = time.perf_counter()
    space.POOL.update(executor=candidate_workers, generation=registries.GENERATION)
    LOG.debug("candidate pool held generation=%d seconds=%.4f", registries.GENERATION, time.perf_counter() - started)
    yield candidate_workers
    space.POOL.clear()
