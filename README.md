sudo apt install python3-dev python3-pip virtualenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
git clone https://github.com/enijkamp/short_run_mcmc_apex.git
cd short_run_mcmc_apex
pip3 install -r requirements.txt
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../short_run_mcmc_apex
python3 short_run_mcmc_apex.py