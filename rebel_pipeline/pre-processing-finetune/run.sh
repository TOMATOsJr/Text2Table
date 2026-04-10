#!/bin/bash
#SBATCH -A research
#SBATCH -J "rebel_train"
#SBATCH -c 36
#SBATCH --mem-per-cpu=2048
#SBATCH -G 4
#SBATCH -o "output.txt"
#SBATCH -e "output.txt"
#SBATCH --time="4-00:00:00"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jayanth.raju@research.iiit.ac.in
#SBATCH -w gnode045

source "/home2/jayanth.raju/rebel_dataset/venv/bin/activate"

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"

torchrun --nproc_per_node=4 finetune_rebel.py   --train-file /home2/jayanth.raju/rebel_dataset/half_dataset/grounded_train_output.jsonl   --valid-file /home2/jayanth.raju/rebel_dataset/half_dataset/grounded_valid_output.jsonl   --test-file /home2/jayanth.raju/rebel_dataset/half_dataset/grounded_test_output.jsonl   --output-dir /home2/jayanth.raju/rebel_dataset/Text2Table/rebel_pipeline/rebel_finetuned_grounded_half
echo "Time at exit: $(date)"
