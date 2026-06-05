import asyncio

async def test_gather():
    async def task1():
        raise KeyError('\n  "name"')
    
    async def task2():
        return {"data": 2}
        
    try:
        results = await asyncio.gather(task1(), task2())
    except Exception as e:
        print("Gather raised:", repr(str(e)))
        
asyncio.run(test_gather())
