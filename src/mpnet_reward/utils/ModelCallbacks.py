from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .save_csv import save_csv
import pandas as pd
from copy import deepcopy

class TrainValCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

class SaveModelOutputsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "eval_pos_rewards" in logs:
            df = pd.DataFrame(
                {
                    "global_step": [state.global_step] * len(logs["eval_pos_rewards"]),
                    "pos_rewards": logs["eval_pos_rewards"],
                    "neg_rewards": logs["eval_neg_rewards"],
                }
            )
            save_csv(df, f"{args.output_dir}/eval_rewards_{args.process_index}.csv")
