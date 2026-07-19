"""Planning: turn a validated intent into a finite, ordered analysis plan.

Deterministic and LLM-free. Targeted mode instantiates a per-analysis_type
template; open-ended mode emits a capped exploration battery ranked off the
profile. The plan is finite by construction, which is what makes graph
done-detection structural (see state.is_done).
"""
