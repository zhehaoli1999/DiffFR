conda activate diffFR
python setup.py bdist_wheel
pip install -I build/dist/*.whl
