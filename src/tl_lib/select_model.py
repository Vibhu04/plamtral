from ...parameter_efficient.mam_adapter import Model_with_mam_adapter
from ...parameter_efficient.prefix_tuning import Model_with_prefix_tuning
from ...parameter_efficient.prompt_tuning import Model_with_prompt_tuning
from ...parameter_efficient.parallel_adapter import Model_with_parallel_adapter
from ...parameter_efficient.adapterdrop import *
from ...parameter_efficient.adapter import Model_with_adapter
from ...parameter_efficient.adapter_bapna import Model_with_adapter_bapna
from ...parameter_efficient.lora import Model_with_LoRA
from utils import get_GPT2LMH


def select_model(base_model, model_size, technique, dropout):

    models = {
      'GPT2LMH': get_GPT2LMH(model_size, dropout),
      'Houlsby Adapter': Model_with_adapter(),
      'Bapna Adapter': Model_with_adapter_bapna(),
      'AdapterDrop Specialised': Model_with_adapterdrop_spec(),
      'AdapterDrop Robust': Model_with_adapterdrop_rob(),
      'Parallel Adapter': Model_with_parallel_adapter(),
      'MAM Adapter': Model_with_mam_adapter(),
      'LoRA': Model_with_LoRA(),
      'Prompt Tuning': Model_with_prompt_tuning(),
      'Prefix Tuning': Model_with_prefix_tuning()
    }

    model = models[technique]

    return model


