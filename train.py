import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid

from TPN.models.solver import Solver
from TPN.models.evaluator import Evaluator
from TPN.readers.rawframes_dataset import RawFramesDataset
from TPN.utils.config import cfg


def evaluate(evaluator, epoch):
    with fluid.dygraph.guard():
        print("evaluating models...")
        evaluator.reset_summary()
        evaluator.load_models(epoch)

        for step, data in evaluator.read_data():
            images, label = data
            loss, accuracy = evaluator.forward(images, label)

            if step % cfg.solver.log_interval == 0:
                evaluator.update_logger(epoch, step, loss, accuracy)

        evaluator.write_summary(epoch)


def train():
    with fluid.dygraph.guard():
        dataset = RawFramesDataset(**cfg.dataset.hmdb)
        batch_reader = dataset.batch_reader(cfg.solver.batch_size)
        solver = Solver(batch_reader)

        eval_dataset = RawFramesDataset(**cfg.eval.dataset.hmdb)
        eval_batch_reader = eval_dataset.batch_reader(cfg.eval.solver.batch_size)
        evaluator = Evaluator(eval_batch_reader)

        if cfg.solver.start_epoch != 0:
            solver.load_models(cfg.solver.start_epoch - 1)

        for epoch in range(cfg.solver.start_epoch, cfg.solver.max_epoch):
            print("Begin to train epoch " + str(epoch))
            for step, data in solver.read_data():
                images, label = data
                loss, accuracy = solver.forward(images, label)
                solver.backward(loss)

                if step % cfg.solver.log_interval == 0:
                    solver.update_logger(epoch, step, loss, accuracy)
            
            if epoch % cfg.solver.save_interval == 0:
                solver.save_models(epoch)
                evaluate(evaluator, epoch)


if __name__ == '__main__':
    train()