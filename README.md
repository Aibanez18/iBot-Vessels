# iDoc

For a glimpse at the full documentation of iBOT pre-training, please run:
```bash
python main_ibot.py --help
   ```
To start the iBOT pre-training with LoRA, simply run the following command:
```bash
python main_ibot.py \
--arch vit_lora \
--data_path /../your_dataset.pkl \
--old_path /../old_path \
--new_path /../new_path \
--ckpt_path_student /../student.pth \
--ckpt_path_teacher /../teacher.pth \
--epochs 50 \
--batch_size_per_gpu 16 \
--output_dir ./output_dir
   ```
