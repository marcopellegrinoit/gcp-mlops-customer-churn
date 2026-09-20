{#
  A bounds test for numeric columns. Written here rather than pulled from dbt_utils because
  that is the project's only would-be package dependency, and taking it would add a
  `dbt deps` step plus a vendored packages lockfile to the container build for one test.

  NULLs pass: absence is the concern of a not_null test, and folding the two together makes
  a failure ambiguous about which guarantee broke.
#}
{% test accepted_range(model, column_name, min_value, max_value) %}

SELECT {{ column_name }}
FROM {{ model }}
WHERE {{ column_name }} IS NOT NULL
  AND ({{ column_name }} < {{ min_value }} OR {{ column_name }} > {{ max_value }})

{% endtest %}
