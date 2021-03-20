## Dependencies
- Python 3.8.8
- PyTorch ==1.7.1
- Pyflow -> [https://github.com/pathak22/pyflow](https://github.com/pathak22/pyflow)
```bash
cd pyflow/
python setup.py build_ext -i
cp pyflow*.so ..
cd ..
```
## Training
```bash
python main.py --HR_dir="PATH/TO/HR" --LR_dir="PATH/TO/LR" --file_list='NTIRE21/train.txt'
```
- You can modify the `gpus_list=[0,1]` in the code to adapt to your device

## Testing
```bash
python test.py --LR_dir="PATH/TO/LR" --file_list='NTIRE21/test.txt'
```
- You can modify the `gpus_list=[0,1]` in the `test.py` to adapt to your device.

- You can modify the `'--model'` parameter in the `test.py` to load different trained model.