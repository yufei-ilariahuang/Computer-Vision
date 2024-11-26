import pyttsx3


def tts_process_function(message_queue):
    tts_engine = pyttsx3.init()
    print("TTS engine initialized in process.")
    while True:
        try:
            message = message_queue.get()
            if message is None:
                break
            tts_engine.say(message)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"Error in TTS process: {e}")


def tts_closest_object(distances, message_queue):
    if distances:  # ensure the distances list is not empty
        valid_distances = [d for d in distances if d is not None and d[0] is not None]
        if valid_distances:
            min_distance = min(valid_distances, key=lambda x: x[0])
            message = (
                f"{min_distance[1]} is {min_distance[0]:.2f} meters away from you."
            )
        else:
            message = "No valid distances available."
        message_queue.put(message)
    else:
        message_queue.put("No object detected currently.")
