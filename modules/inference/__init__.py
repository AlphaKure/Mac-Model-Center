from asyncio import AbstractEventLoop, Queue, Event

from schemas.inference import Progress

def _callback(
    pipeline,
    step: int|None = None, 
    timestep= None,
    callback_kwargs= None,
    *,
    stopEvent: Event,
    loop: AbstractEventLoop,
    queue: Queue,
    totalSteps: int,
):
    """for pipeline return progress"""

    if step is not None:
        percentage = round((int(step)+1)/totalSteps*100,2)
        if stopEvent.is_set():
            pipeline._interrupt = True
            loop.call_soon_threadsafe(queue.put_nowait,Progress(result= "Interrupt", percentage= percentage, status= "running").dict())
        else:
            loop.call_soon_threadsafe(queue.put_nowait,Progress(result= "", percentage= percentage, status= "running").dict())

    return callback_kwargs