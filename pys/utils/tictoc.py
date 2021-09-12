import time

class TicToc():
  def __init__(self):
    self.start = time.time()
    self.end = time.time()

  def tic(self):
    self.start = time.time()

  def toc(self):
    self.end = time.time()

  def get_dt(self):
    return self.end - self.start

if __name__ == "__main__":
  timer1 = TicToc()
  timer2 = TicToc()
  for _ in range(5):
    timer1.tic()
    for _ in range(5):
      timer2.tic()
      time.sleep(0.5)
      timer2.toc()
      print('timer2 : {:8.4f}'.format(timer2.get_dt()))
    timer1.toc()
    print('timer1 : {:8.4f}'.format(timer1.get_dt()))
  print('end')
