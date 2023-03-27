def summary_writer(experiment_name, model_name, extra=None):
  from datetime import datetime
  import os
  from torch.utils.tensorboard.writer import SummaryWriter

  timestamp = datetime.now().strftime('%Y-%m-%d')

  if extra:
    log_dir = os.path.join('runs', timestamp, experiment_name, model_name, extra)
  else:
    log_dir = os.path.join('runs', timestamp, experiment_name, model_name)

  return SummaryWriter(log_dir)