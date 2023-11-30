.. raw:: html

    <div class="api-module">

.. currentmodule:: {{ module }}


{{ fullname | underline }}



.. automodule:: {{ fullname }}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
  :toctree: ./
{% for item in classes %}
  {{ fullname }}.{{ item }}
{% endfor %}
{% endif %}
{% endblock %}


{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
  :toctree: ./
{% for item in functions %}
  {{ fullname }}.{{ item }}
{% endfor %}
{% endif %}
{% endblock %}


{% block exceptions %}
{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
  :toctree: ./
{% for item in exceptions %}
  {{ fullname }}.{{ item }}
{% endfor %}
{% endif %}
{% endblock %}

.. raw:: html

    </div>
