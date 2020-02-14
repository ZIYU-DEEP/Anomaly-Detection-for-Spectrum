cd ~/Anomaly-Detection-for-Spectrum/code/preprocessing

python spec2rss.py downtown normal
python spec2rss.py downtown_LOS-5M-USRP1 abnormal
python spec2rss.py downtown_LOS-5M-USRP2 abnormal
python spec2rss.py downtown_LOS-5M-USRP3 abnormal
python spec2rss.py downtown_NLOS-5M-USRP1 abnormal
python spec2rss.py downtown_Dynamics-5M-USRP1 abnormal

python spec2rss.py campus_drive normal
python spec2rss.py campus_drive_LOS-5M-USRP1 abnormal
python spec2rss.py campus_drive_LOS-5M-USRP2 abnormal
python spec2rss.py campus_drive_LOS-5M-USRP3 abnormal
python spec2rss.py campus_drive_NLOS-5M-USRP1 abnormal
python spec2rss.py campus_drive_Dynamics-5M-USRP1 abnormal

cd /net/adv_spectrum
chmod 777 -R ./
