import sounddevice as sd
import numpy as np

import whisper

import asyncio
import queue
import sys

import pulsectl


# SETTINGS
MODEL_TYPE="small.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE="English"
# pre-set the language to avoid autodetection
BLOCKSIZE=24678 
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds. 
SILENCE_THRESHOLD=400
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO=200
# number of samples in one buffer that are allowed to be higher than threshold


global_speaker_ndarray = None
global_mic_ndarray = None
model = whisper.load_model(MODEL_TYPE)

pulse = pulsectl.Pulse('whisper-client')
pulse_monitor_name = pulse.server_info().default_sink_name + '.monitor'
pulse_monitor = [i for i in pulse.source_list() if i.name == pulse_monitor_name][-1]

pulse_source_name = pulse.server_info().default_source_name
pulse_source = [i for i in pulse.source_list() if i.name == pulse_source_name][-1]

async def speakerstream_generator():
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
    pulse_output = [o for o in pulse.source_output_list() if o.name == 'ALSA Capture'][-1]
    pulse.source_output_move(pulse_output.index, pulse_monitor.index)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status
            
async def micstream_generator():
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
    pulse_output = [o for o in pulse.source_output_list() if o.name == 'ALSA Capture'][-1]
    pulse.source_output_move(pulse_output.index, pulse_source.index)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status
            
async def process_speaker_buffer():
    global global_speaker_ndarray
    async for indata, status in speakerstream_generator():
        
        indata_flattened = abs(indata.flatten())
                
        # discard buffers that contain mostly silence
        if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO):
            continue
        
        if (global_speaker_ndarray is not None):
            global_speaker_ndarray = np.concatenate((global_speaker_ndarray, indata), dtype='int16')
        else:
            global_speaker_ndarray = indata
            
        # concatenate buffers if the end of the current buffer is not silent
        if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/15):
            continue
        else:
            local_ndarray = global_speaker_ndarray.copy()
            global_speaker_ndarray = None
            indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
            result = model.transcribe(indata_transformed, language=LANGUAGE)
            print("speaker" + result["text"])
            
        del local_ndarray
        del indata_flattened

async def process_mic_buffer():
    global global_mic_ndarray
    async for indata, status in micstream_generator():
        
        indata_flattened = abs(indata.flatten())
                
        # discard buffers that contain mostly silence
        if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO):
            continue
        
        if (global_mic_ndarray is not None):
            global_mic_ndarray = np.concatenate((global_mic_ndarray, indata), dtype='int16')
        else:
            global_mic_ndarray = indata
            
        # concatenate buffers if the end of the current buffer is not silent
        if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/15):
            continue
        else:
            local_ndarray = global_mic_ndarray.copy()
            global_mic_ndarray = None
            indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
            result = model.transcribe(indata_transformed, language=LANGUAGE)
            print("mic:" + result["text"])
            
        del local_ndarray
        del indata_flattened


async def main():
    print('\nActivating wire ...\n')
    speaker_task = asyncio.create_task(process_speaker_buffer())
    mic_task = asyncio.create_task(process_mic_buffer())
    while True:
        await asyncio.sleep(1)
    speaker_task.cancel()
    mic_task.cancel()
    try:
        await audio_task
    except asyncio.CancelledError:
        print('\nwire was cancelled')


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
