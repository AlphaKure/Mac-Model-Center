from asyncio import AbstractEventLoop, Queue

from schemas.inference import Progress

def _callback(loop: AbstractEventLoop, queue: Queue, totalSteps: int, *args, **kargs):
    """for pipeline return progress"""
    
    if len(args) == 4:
        __ , step, __, callback_kwargs = args
    elif len(args) == 3:
        step, __, callback_kwargs = args
    if step is not None:
        percentage = round((int(step)+1)/totalSteps*100,2)
        loop.call_soon_threadsafe(queue.put_nowait,Progress(result= "", percentage= percentage, status= "running").dict())
    return callback_kwargs if isinstance(callback_kwargs, dict) else {}