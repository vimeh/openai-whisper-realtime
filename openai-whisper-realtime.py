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

global_ndarrays = {} 
model = whisper.load_model(MODEL_TYPE)

pulse = pulsectl.Pulse('whisper-client')

async def inputstream_generator(pulse_idx):
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
    pulse_output = [o for o in pulse.source_output_list() if o.name == 'ALSA Capture'][-1]
    pulse.source_output_move(pulse_output.index, pulse_idx)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status
            
async def process_audio_buffer(pulse_idx):
    global global_ndarrays 
    async for indata, status in inputstream_generator(pulse_idx):
        
        indata_flattened = abs(indata.flatten())
                
        # discard buffers that contain mostly silence
        if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO):
            continue
        
        if (global_ndarrays[pulse_idx] is not None):
            global_ndarrays[pulse_idx] = np.concatenate((global_ndarrays[pulse_idx], indata), dtype='int16')
        else:
            global_ndarrays[pulse_idx] = indata
            
        # concatenate buffers if the end of the current buffer is not silent
        if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/15):
            continue
        else:
            local_ndarray = global_ndarrays[pulse_idx].copy()
            global_ndarrays[pulse_idx] = None
            indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
            result = model.transcribe(indata_transformed, language=LANGUAGE)
            print(f"{pulse_idx}: {result['text']}")
            
        del local_ndarray
        del indata_flattened

async def main():
    print('\nActivating wire ...\n')
    pulse_speaker_name = pulse.server_info().default_sink_name + '.monitor'
    pulse_speaker = pulse.get_source_by_name(pulse_speaker_name) 

    pulse_microphone_name = pulse.server_info().default_source_name
    pulse_microphone = pulse.get_source_by_name(pulse_microphone_name) 

    pulse_idxs = [pulse_speaker.index, pulse_microphone.index]
    tasks= []
    for pulse_idx in pulse_idxs:
        global_ndarrays[pulse_idx] = None
        tasks.append(asyncio.create_task(process_audio_buffer(pulse_idx)))
    while True:
        await asyncio.sleep(1)
    for task in taskpool:
        task.cancel()
    try:
        for task in tasks:
            await task
    except asyncio.CancelledError:
        print('\nwire was cancelled')


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
