"""## Models for the Speclet project.

### How to add a new model:

1. Subclass `src.models.speclet_model.SpecletModel`.
2. Update the `build_model()` method.
3. Describe the model in the new class's docstring.
4. Add tests.
5. Add the model to the following locations:
    - **src/models/configuration.py**: to the `model_option_map` in `get_model_class()`
    - **src/project_enums.py**: to the `ModelOption` enum
    - **src/types.py**: to `SpecletProjectModelTypes` and `ModelConfigurationT` if the
      model has a custom configuration class
6. Add the model to the model configuration file **models/model-configs.yaml**.
7. Check the configuration file format with `make check_model_config`.

A good and simple example is the `src.models.speclet_simple.SpecletSimple` model and its
tests.
"""
