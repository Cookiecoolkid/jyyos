**lockdep**: 为每一个锁都追踪上锁的顺序会带来相当的开销。更经济的方式是把所有在同一行代码中初始化的锁都看成是 “同一个锁”，这样锁的数量就大幅减少了。当然这也会损失一些检测精度，例如哲学家吃饭问题中，如果所有的锁都在同一行代码中初始化，我们就不能区分它们的上锁顺序了。