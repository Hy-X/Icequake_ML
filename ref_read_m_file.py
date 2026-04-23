import matplotlib.pyplot as plt
import numpy as np
import obspy
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from obspy.imaging.spectrogram import spectrogram  # function, not a Trace method
from obspy import read_events
from obspy.core.event import Comment
import glob

# Read data (raw string for Windows path)
cat = read_events("filtered_events.xml")
files = glob.glob('/home/student/Desktop/top300/raw_cut_waveforms/*.m')

evcount = 0
snr_s = []
for ev in cat:
    print(ev)
    ot = ev.preferred_origin().time
    file = glob.glob('/home/student/Desktop/top300/raw_cut_waveforms/'+str((ot.year))+str(ot.month).zfill(2)+str(ot.day).zfill(2)+str(ot.hour).zfill(2)+str(ot.minute).zfill(2)+str(ot.second).zfill(2)+'*.m')
    print(file)
    try:
        file = file[0]
        print(file)
    
        print(file)
        st = obspy.read(file)
        for i, tr in enumerate(st):
            print(f"Plotting trace {i}: {tr.stats.station}.{tr.stats.channel}")
       
            # prep & filtering
            sr = tr.stats.sampling_rate
            tr_filt = tr.copy()
            tr_filt.filter('highpass', freq=5.0, corners=4, zerophase=True)
            tr_filt.filter('bandpass', freqmin=20.0, freqmax=80.0, corners=2, zerophase=True)
       
            # STA/LTA & triggers (use filtered data)
            cft = classic_sta_lta(tr_filt.data, int(0.5 * sr), int(10.0 * sr))
            trigs = trigger_onset(cft, thres1=2.0, thres2=1.0)
       
            # figure/axes
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # waveform
            ax1.set_ylabel("Filtered")
            # ax1.set_title(f"{tr.stats.station}.{tr.stats.channel}")
            ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.6], sharex=ax1)  # spectrogram
            ax2.set_ylabel("Frequency [Hz]")
            ax2.set_xlabel("Time [s]")   
       
            # time vector in seconds
            n = tr_filt.stats.npts
            t = np.arange(n) / sr
       
            # --- add UTCdatetime ---
            start = tr.stats.starttime
            # ax1.set_title(f"{tr.stats.station}.{tr.stats.channel} - {start}")
       
            # plot waveform
            ax1.plot(t[40:], tr_filt.data[40:], 'k')
       
            # plot spectrogram (function form)
            # wlen is window length in seconds
            im = spectrogram(tr_filt.data[40:], samp_rate=sr, wlen=2.0,
                             log=False, axes=ax2, show=False)
       
            # Calculate ymax for text annotations
            ymax = 0.9 * np.max(np.abs(tr_filt.data)) if np.max(np.abs(tr_filt.data)) else 1.0
           
            # add picks (use seconds, not datetimes)
            for pick in ev.picks:
                # print(pick)
                if pick.waveform_id.station_code == tr.stats.station:
                    if pick.phase_hint == 'P':
                        pick_sec = float(pick.time - tr.stats.starttime)
                        ax1.axvline(x=pick_sec, linestyle=':', linewidth=1.2, color='g', zorder=10)
         
                    elif pick.phase_hint == 'S':
                        pick_sec = float(pick.time - tr.stats.starttime)
                        ax1.axvline(x=pick_sec, linestyle=':', linewidth=1.2, color='g', zorder=10)
                   
                    snr = np.sum(np.abs(tr_filt.copy().trim(pick.time-0.25,pick.time+0.25)))/np.sum(np.abs(tr_filt.copy().trim(pick.time-2.25,pick.time-1.75)))
                    # ^ add this to geowriting project
                    print(snr)
                    snr_s.append(snr)
                    pick.comments.append(Comment(text=str(snr)))
                   
                    # UNCOMMENT THIS TO SAVE IMAGES
            #plt.show()
            #plt.savefig(f"{tr.stats.station}.{tr.stats.channel}.{evcount}.png")
           
            if len(trigs) > 0:
                # first trigger window
                p_sample = trigs[0][0]
                s_sample = trigs[0][1]
                p_sec = p_sample / sr
                s_sec = s_sample / sr
                ax1.axvline(x=p_sec, linestyle='--', linewidth=1.8, color='b', zorder=20)
                ax1.text(p_sec, ymax, 'P', color='b', fontsize=11, ha='center', va='top',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=21)
       
                ax1.axvline(x=s_sec, linestyle='--', linewidth=1.8, color='r', zorder=20)
                ax1.text(s_sec, ymax, 'S', color='r', fontsize=11, ha='center', va='top',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=21)
       
            ax1.relim(); ax1.autoscale_view()
            
            ax1.set_title(f"{tr.stats.station}.{tr.stats.channel} - {start}, {snr}")
    
            plt.show()
    except:
        pass
    evcount =+ 1


print(evcount)