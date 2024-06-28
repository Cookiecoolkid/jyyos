# jyyos

## Project code is taken from NJU-2024-operation-system by jyy.
## Please check [https://jyywiki.cn/OS/2024/](https://jyywiki.cn/OS/2024/)

## 下面仅展示部分项目的部分代码及部分解释，完整项目及代码请查看源码

## logisim ⭐️⭐️

- 本示例用代码模拟 logisim 数字电路的功能. 详细代码见 *introduction* 分支下的 *logisim* 
- logisim 中的主要用到 *reg*, *wire* 类型的变量：
```c
// Wires
// A wire in digital circuits is used to connect different components
// together, allowing the transmission of electrical signals between them.
// A wire is represented as a boolean value (true or false), which
// corresponds to high (1) and low (0) voltage levels in a real circuit.
typedef bool wire;

// Flip-flops
// A basic memory element in digital circuits, capable of storing one bit
// of data. It has an input and an output (wires), and it maintains its
// output value until the input changes and a clock signal is received.
typedef struct {
    bool value;  // The current value stored in the flip-flop
    wire *in;    // Pointer to the input wire
    wire *out;   // Pointer to the output wire
} reg;
```

*wire* 被定义为 *bool* 类型，表示电路中的高低电压，*reg* 用于存储一个比特的数据，有输入输出端口.

而数字电路中还需要有逻辑门，定义如下：
```c
// Logical gates from NAND
// NAND gate is a fundamental building block in digital electronics. Using
// NAND gates, one can construct any other logical operation.

// NAND gate: Returns true unless both inputs are true.
#define NAND(X, Y)  (!((X) && (Y)))

// NOT gate: Inverts the input.
#define NOT(X)      (NAND(X, 1))

// AND gate: Returns true only if both inputs are true.
#define AND(X, Y)   (NOT(NAND(X, Y)))

// OR gate: Returns true if at least one input is true.
#define OR(X, Y)    (NAND(NOT(X), NOT(Y)))
```

Code:
```c
    X1 = AND(NOT(X), Y);
    Y1 = NOT(OR(X, Y));
    A = D = E = NOT(Y);
    B = 1;
    C = NOT(X);
    F = Y1;
    G = X;

    // 2. Edge triggering: Lock values in the flip-flops
    b0.value = *b0.in;
    b1.value = *b1.in;
    *b0.out = b0.value;
    *b1.out = b1.value;
```
每当时钟周期到来的时候，将输入的值存储到 *reg* 中，然后输出到 *wire* 中，各个*wire*的值就会根据逻辑门的定义进行计算.

由此，某种意义上，计算机系统就是可以直接由此构造出来.

> Everything is a State Machine.

只需要用数字电路模拟出如下部件:
- REG
- CSR
- Memory
就可以完整表示一个 CPU State. 而每一次需要计算下一步状态即可，这就是一个小的 CPU 模型.
可以见下一节 *mini-rv32ima* 的代码.

## mini-rv32ima ⭐️⭐️⭐️

本节代码是一个简单的 RISC-V CPU 模型，详细代码见 *introduction* 分支下的 *mini-rv32ima*

执行如下命令

```bash
$ make mini-rv32ima
$ ./mini-rv32ima ./bin/fib.rv32i-bin 10
```

即可在 *a0* 寄存器中得到斐波那契数列的第 10 项数值.

*CPU-State* 的定义如下:

```c
struct CPUState {
    // Processor internal state
    uint32_t regs[32], csrs[CSR_COUNT];

    // Memory state
    uint8_t *mem;
    uint32_t mem_offset, mem_size;
};
```
*REG* 和 *CSR* 用枚举类型表示:

```c
enum RV32IMA_REG {...};
enum RV32IMA_CSR {...};
```

> Everything is a State Machine. :smile:

因此，只需要定义好 *CPU-State* 的结构，然后定义好 *CPU* 的状态转移函数即可.

```c
static inline int32_t rv32ima_step(struct CPUState *state, uint32_t elapsedUs);
```

其中每一次 *step* 函数的调用，就是一个时钟周期的过程，即一个状态的转移.
共可能有以下几种状态转移:
- Traps
- Timer interrupts
- run instructions

Traps 和 Timer interrupts 都是通过修改 *CSR* 来实现状态迁移：

```c
cycle_end:
    // Handle traps and interrupts.
    if (trap) {
        if (trap & 0x80000000) { // It's an interrupt, not a trap.
            CSR(MCAUSE) = trap;
            CSR(MTVAL) = 0;
            pc += 4; // PC needs to point to where the PC will return to.
        } else {
            CSR(MCAUSE) = trap - 1;
            CSR(MTVAL) = (trap > 5 && trap <= 8) ? rval : pc;
        }
        CSR(MEPC) = pc;
        // On an interrupt, the system moves current MIE into MPIE
        CSR(MSTATUS) = ((CSR(MSTATUS) & 0x08) << 4) | ((CSR(EXTRAFLAGS) & 3) << 11);
        pc = (CSR(MTVEC) - 4);

        // If trapping, always enter machine mode.
        CSR(EXTRAFLAGS) |= 3;

        trap = 0;
        pc += 4;
    }

    if (CSR(CYCLEL) > cycle)
        CSR(CYCLEH)++;
    CSR(CYCLEL) = cycle;
    CSR(PC) = pc;
    return 0;
```

执行指令的过程即抽象为如下代码：


```c
    // 1. Fetch instruction
    uint32_t instr = rv32ima_fetch(state);

    // 2. Decode instruction
    struct RV32IMA_DecodeResult decode = rv32ima_decode(instr);

    // 3. Execute instruction
    struct RV32IMA_ExecResult exec = rv32ima_exec(state, decode);

    // 4. Write back result
    rv32ima_writeback(state, decode, exec);
```

## minimal ⭐️⭐️⭐️

- 平时所写的应用程序通过编译汇编链接后，所形成的可执行文件用 *objdump* 查看，可以看到其中链接了很多的代码，那么如何实现一个最小的可执行文件？
- 指令集并没有提供推出应用程序的指令，应用程序如何退出？

答案就在于**syscall**指令.

*minimal.S* 的代码如下：

```c
#include <sys/syscall.h>

// The x86-64 system call Application Binary Interface (ABI):
//     System call number: RAX
//     Arguments: RDI, RSI, RDX, RCX, R8, R9
//     Return value: RAX
// See also: syscall(2) syscalls(2)

#define syscall3(id, a1, a2, a3) \
    movq $SYS_##id, %rax; \
    movq $a1, %rdi; \
    movq $a2, %rsi; \
    movq $a3, %rdx; \
    syscall

#define syscall2(id, a1, a2)  syscall3(id, a1, a2, 0)
#define syscall1(id, a1)  syscall2(id, a1, 0)

.globl _start
_start:
    syscall3(write, 1, addr1, addr2 - addr1)
    syscall1(exit, 1)

addr1:
    .ascii "\033[01;31mHello, OS World\033[0m\n"
addr2:
```

输入如下命令后：
    
```bash
$ make minimal
$ objdump -d minimal
```
得到：
    
```bash
    
minimal:     file format elf64-x86-64


Disassembly of section .text:

0000000000401000 <_start>:
  401000:	48 c7 c0 01 00 00 00 	mov    $0x1,%rax
  401007:	48 c7 c7 01 00 00 00 	mov    $0x1,%rdi
  40100e:	48 c7 c6 3c 10 40 00 	mov    $0x40103c,%rsi
  401015:	48 c7 c2 1c 00 00 00 	mov    $0x1c,%rdx
  40101c:	0f 05                	syscall 
  40101e:	48 c7 c0 3c 00 00 00 	mov    $0x3c,%rax
  401025:	48 c7 c7 01 00 00 00 	mov    $0x1,%rdi
  40102c:	48 c7 c6 00 00 00 00 	mov    $0x0,%rsi
  401033:	48 c7 c2 00 00 00 00 	mov    $0x0,%rdx
  40103a:	0f 05                	syscall 

000000000040103c <addr1>:
  40103c:	1b 5b 30             	sbb    0x30(%rbx),%ebx
  40103f:	31 3b                	xor    %edi,(%rbx)
  401041:	33 31                	xor    (%rcx),%esi
  401043:	6d                   	insl   (%dx),%es:(%rdi)
  401044:	48                   	rex.W
  401045:	65 6c                	gs insb (%dx),%es:(%rdi)
  401047:	6c                   	insb   (%dx),%es:(%rdi)
  401048:	6f                   	outsl  %ds:(%rsi),(%dx)
  401049:	2c 20                	sub    $0x20,%al
  40104b:	4f 53                	rex.WRXB push %r11
  40104d:	20 57 6f             	and    %dl,0x6f(%rdi)
  401050:	72 6c                	jb     4010be <addr2+0x66>
  401052:	64 1b 5b 30          	sbb    %fs:0x30(%rbx),%ebx
  401056:	6d                   	insl   (%dx),%es:(%rdi)
  401057:	0a                   	.byte 0xa
```

可以看到，*minimal* 可执行文件中只有两个系统调用，一个是 *write* 一个是 *exit*. 这就是最小的可执行文件:smile:

### strace

事实上，任意一个程序本质上都是和 *minimal* 一样，都是状态的迁移以及 *syscall* 的调用.
而 *strace* 命令可以很好查看程序的系统调用，如下：

```bash
$ strace ./minimal
```

## hanoi ⭐️⭐️⭐️

- 汉诺塔问题是一个经典的递归问题，但是如何将递归的问题转化为迭代的问题？

> Everything is a State Machine.

C 程序本质也是一个状态机：
- 状态由变量数据与栈帧组成

那么非递归的汉诺塔的状态即为：
```c
struct Frame {
    // Each frame has a program counter to keep track its next
    // to-be-executed statement.
    int pc;

    // The internal state of the frame. This state includes
    // both arguments and local variables (if any).
    //
    // Arguments:
    int n;
    char from, to, via;

    // Local variables:
    int c1, c2;
};
```
- 每一个栈帧都有自己的 *pc* 记录下一步的执行指令(即会记录函数返回后的下一步指令地址)
- 每一个栈帧有自己的变量数值，包括参数和局部变量.

那么函数调用实际上就是新增一个栈帧，函数返回就是去除一个栈帧：

```c
int hanoi(int n, char from, char to, char via) {
    Frame stk[64];
    Frame *top = stk - 1;

    // Function call: push a new frame (PC=0) onto the stack
    #define call(...) ({ *(++top) = (Frame){.pc = 0, __VA_ARGS__}; })
    
    // Function return: pop the top-most frame
    #define ret(val) ({ top--; retval = (val); })


    // The last function-return's value. It is not obvious
    // that we only need one retval.
    int retval = 0;

    // The initial call to the recursive function
    call(n, from, to, via);

    while (1) {
        // Fetch the top-most frame.
        Frame *f = top;
        if (top < stk) {
            // No top-most frame any more; we're done.
            break;
        }

        // Jumps may change this default next pc.
        int next_pc = f->pc + 1;

        // Single step execution.

        // Extract the parameters from the current frame. (It's
        // generally a bad idea to reuse variable names in
        // practice; but we did it here for readability.)
        int n = f->n, from = f->from, to = f->to, via = f->via;

        switch (f->pc) {
            case 0:
                if (n == 1) {
                    printf("%c -> %c\n", from, to);
                    ret(1);
                }
                break;
            case 1: call(n - 1, from, via, to); break;
            case 2: f->c1 = retval; break;
            case 3: call(1, from, to, via); break;
            case 4: call(n - 1, via, to, from); break;
            case 5: f->c2 = retval; break;
            case 6: ret(f->c1 + f->c2 + 1); break;
            default: assert(0);
        }

        f->pc = next_pc;
    }

    return retval;
}
```

### 状态机视角下的编译器

- 高级语言(C) = 状态机 (栈 + 变量数值)
- 汇编代码(汇编指令) = 状态机 (REG + CSR + Memory)
- 编译器 = 状态机之间的翻译器

### 状态机视角下的编译优化

只要编译前与编译后输出的系统调用序列完全一致，那么编译器做的优化就是正确的.

## os-model ⭐️⭐️⭐️

在本实验之下，用三十多行 *python* 代码实现了一个简单的操作系统模型，其中只包含*spawn*，*read*，*write* 三个系统调用.

- *spawn*：创建一个新的进程
- *read*：读取一个字符
- *write*：写入一个字符

其中用到了 *generator* 的概念，即每一个进程都是一个 *generator* 对象，每一次调用 *step* 函数，都会执行到下一个系统调用.

此简易的模型就包含了操作系统最重要的特性：
- 进程
- 系统调用
- 上下文切换
- 调度

```python
#!/usr/bin/env python3

import sys
import random
from pathlib import Path

class OS:
    SYSCALLS = ['read', 'write', 'spawn']

    class Process:
        def __init__(self, func, *args):
            # func should be a generator function. Calling
            # func(*args) returns a generator object.
            self._func = func(*args)

            # This return value is set by the OS's main loop.
            self.retval = None

        def step(self):
            '''
            Resume the process with OS-written return value,
            until the next system call is issued.
            '''
            syscall, args, *_ = self._func.send(self.retval)
            self.retval = None
            return syscall, args

    def __init__(self, src):
        # This is a hack: we directly execute the source
        # in the current Python runtime--and main is thus
        # available for calling.
        exec(src, globals())
        self.procs = [OS.Process(main)]
        self.buffer = ''

    def run(self):
        while self.procs:
            current = random.choice(self.procs)

            try:
                # Operating systems handle interrupt and system
                # calls, and "assign" CPU to a process.
                match current.step():
                    case 'read', _:
                        current.retval = random.choice([0, 1])
                    case 'write', s:
                        self.buffer += s
                    case 'spawn', (fn, *args):
                        self.procs += [OS.Process(fn, *args)]
                    case _:
                        assert 0

            except StopIteration:
                # The generator object terminates.
                self.procs.remove(current)

        return self.buffer

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} file')
        exit(1)

    src = Path(sys.argv[1]).read_text()

    # Hack: patch sys_read(...) -> yield "sys_read", (...)
    for syscall in OS.SYSCALLS:
        src = src.replace(f'sys_{syscall}',
                          f'yield "{syscall}", ')

    stdout = OS(src).run()
    print(stdout)
```

## mosiac(model-checker) ⭐️⭐️
实现较复杂，详细代码见 *introduction* 分支下的 *mosaic*.

由于操作系统的存在，不同进度的随机调度以及 IO，使得操作系统的正确性难以验证，因此需要一种模型检测的方法. *mosaic* 就是一个简单的模型检测器，通过将程序的运行过程抽象为 *graph*，然后通过 *BFS* 遍历所有可能的状态，来验证程序的正确性.

目前其包含的系统调用有：

```python
## 1. Mosaic system calls

### 1.1 Process, thread, and context switching

sys_fork = lambda: os.sys_fork()
sys_spawn = lambda fn, *args: os.sys_spawn(fn, *args)
sys_sched = lambda: os.sys_sched()

### 1.2 Virtual character device

sys_choose = lambda choices: os.sys_choose(choices)
sys_write = lambda *args: os.sys_write(*args)

### 1.3 Virtual block storage device

sys_bread = lambda k: os.sys_bread(k)
sys_bwrite = lambda k, v: os.sys_bwrite(k, v)
sys_sync = lambda: os.sys_sync()
sys_crash = lambda: os.sys_crash()
```
其输出结果是 *graph* 中的 *vertice* 和 *edge*，即每一个状态以及状态之间的转移的一个类似 json 的输出，可以将其 *pipe* 给各种工具得到更好的可读性以及应用.


## thread-lib ⭐️⭐️⭐️

```c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>

#define LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))

enum {
    T_FREE = 0, // This slot is not used yet.
    T_LIVE,     // This thread is running.
    T_DEAD,     // This thread has terminated.
};

struct thread {
    int id;  // Thread number: 1, 2, ...
    int status;  // Thread status: FREE/LIVE/DEAD
    pthread_t thread;  // Thread struct
    void (*entry)(int);  // Entry point
};

static struct thread threads_[4096];
static int n_ = 0;

// This is the entry for a created POSIX thread. It "wraps"
// the function call of entry(id) to be compatible to the
// pthread library's requirements: a thread takes a void *
// pointer as argument, and returns a pointer.
static inline
void *wrapper_(void *arg) {
    struct thread *t = (struct thread *)arg;
    t->entry(t->id);
    return NULL;
}

// Create a thread that calls function fn. fn takes an integer
// thread id as input argument.
static inline
void create(void *fn) {
    assert(n_ < LENGTH(threads_));

    // Yes, we have resource leak here!
    threads_[n_] = (struct thread) {
        .id = n_ + 1,
        .status = T_LIVE,
        .entry = fn,
    };
    pthread_create(
        &(threads_[n_].thread),  // a pthread_t
        NULL,  // options; all to default
        wrapper_,  // the wrapper function
        &threads_[n_] // the argument to the wrapper
    );
    n_++;
}

// Wait until all threads return.
static inline
void join() {
    for (int i = 0; i < LENGTH(threads_); i++) {
        struct thread *t = &threads_[i];
        if (t->status == T_LIVE) {
            pthread_join(t->thread, NULL);
            t->status = T_DEAD;
        }
    }
}

__attribute__((constructor)) 
static void startup() {
    atexit(join);
}
```
这段代码是对线程的简易封装，提供了创建线程和等待线程结束的接口：
- `create`：创建一个线程，传入一个函数指针，函数指针的参数是线程的id
- `join`：等待所有线程结束

## Peterson 算法 ⭐️⭐️

```c
#define FALSE 0
#define TRUE 1
#define N 2 // 线程数量

int turn;
int flag[N] = {FALSE, FALSE};

void enter_region(int process) { // process是线程编号，0或1
    int other = 1 - process; // 计算另一个线程的编号
    flag[process] = TRUE; // 表明意图
    turn = other; // 让步
    while (flag[other] == TRUE && turn == other); // 等待
}

void leave_region(int process) { // 离开临界区
    flag[process] = FALSE; // 清除意图
}
```

- 适用于两个线程的互斥问题
- 先举起自己的旗子，表明自己要进入临界区，然后让步，若是观察到对方没有举起旗子或者轮到自己了(`turn != other`)，则进入临界区


## atomic_xchg 自旋锁 ⭐️⭐️⭐️

- 若是只有原子的读和写，是很难实现互斥的.
- 因此硬件提供了`xchg`指令，可以原子地进行一步读和写(交换)
  
```c
int status = ✅;

void lock() {
retry:
    int got = atomic_xchg(&status, ❌);
    if (got != ✅) {
        goto retry;
    }
}

void unlock() {
    atomic_xchg(&status, ✅);
}
```

## 内核中的自旋锁 ⭐️⭐️⭐️

```c
void spin_lock(spinlock_t *lk) {
    // Disable interrupts to avoid deadlock.
    push_off();

    // This is a deadlock.
    if (holding(lk)) {
        panic("acquire %s", lk->name);
    }

    // This our main body of spin lock.
    int got;
    do {
        got = atomic_xchg(&lk->status, LOCKED);
    } while (got != UNLOCKED);

    lk->cpu = mycpu;
}

void spin_unlock(spinlock_t *lk) {
    if (!holding(lk)) {
        panic("release %s", lk->name);
    }

    lk->cpu = NULL;
    atomic_xchg(&lk->status, UNLOCKED);

    pop_off();
}
```
- 此处 push_off 和 pop_off 是每个 CPU 各自记录中断状态的函数
- 当一个 CPU 拿到锁时，会将中断关闭，以避免死锁
- 当此 CPU 所有的锁都释放后(均 `pop`)，才会将中断打开

## 应用程序的互斥 ⭐️⭐️

- 当一个应用程序持有锁的时候，其他任何想获得这把锁的应用程序都会自旋，此时若该应用程序发生了中断，其他应用程序会一直自旋浪费资源.
- 试图将这把锁"让"出去，让其他应用程序在自己中断时有机会获得这把锁.

此时`lock`有两种情况：
- Fast Path: 自旋一次成功得到锁，进入临界区.
- Slow Path: 自旋一次失败，请求`syscall`系统调用 `futex`，将自己挂起，等待锁释放.


## 生产者消费者问题(条件变量) ⭐️⭐️⭐️⭐️

```c
#include <thread.h>
#include <thread-sync.h>

int n, depth = 0;
mutex_t lk = MUTEX_INIT();
cond_t cv = COND_INIT();
 
#define CAN_PRODUCE (depth < n)
#define CAN_CONSUME (depth > 0)

void T_produce() {
    while (1) {
        mutex_lock(&lk);

        while (!CAN_PRODUCE) {
            cond_wait(&cv, &lk);
            // We are here if the thread is being waked up, with
            // the mutex being acquired. Then we check once again,
            // and move out of the loop if CAN_PRODUCE holds.
        }

        // We still hold the mutex--and we check again.
        assert(CAN_PRODUCE);

        printf("(");
        depth++;

        cond_broadcast(&cv);
        mutex_unlock(&lk);
    }
}

void T_consume() {
    while (1) {
        mutex_lock(&lk);

        while (!CAN_CONSUME) {
            cond_wait(&cv, &lk);
        }

        printf(")");
        depth--;

        cond_broadcast(&cv);
        mutex_unlock(&lk);
    }
}
```
- `producer`和`consumer`线程运行的条件不同
  - `CAN_PRODUCE`：`depth < n`
  - `CAN_CONSUME`：`depth > 0`
- 若要正确实现同步：
  - 使用两个条件变量`cv`和`lk`.(但更复杂，`signal`和`broadcast`的使用也需要甄别，易出现 bug)
  - 或者就是如上述代码展示，只用一个条件变量`cv`和一个互斥锁`lk`与`broadcast`相配合(并且要注意使用的是 `while` 循环)，但这样会有稍微降低性能，因为`broadcast`会唤醒所有线程，而`signal`只唤醒一个线程.

## 信号量 ⭐️⭐️⭐️

- 能计数的互斥锁.
- `P`操作：`wait`(`acquire`)，`V`操作：`signal`(`release`)
- 当信号量的数量为 1 时，就是互斥锁
- 信号量 80% 的应用场景是相当于互斥锁(但可以在一个线程获取，另一个线程释放，而互斥锁只能在同一个线程获取和释放)
- 信号量 20% 的应用场景是用于控制资源的数量，如线程池(进行计数)

### 信号量实现生产者消费者问题

```c
#include <thread.h>
#include <thread-sync.h>

sem_t fill, empty;

void T_produce() {
    while (1) {
        // Needs an empty slot for producing.
        P(&empty);

        printf("(");

        // Creates a filled slot.
        V(&fill);
    }
}

void T_consume() {
    while (1) {
        // Needs a filled slot for consuming.
        P(&fill);

        printf(")");
        
        // Creates an empty slot.
        V(&empty);
    }
}

int main(int argc, char *argv[]) {
    assert(argc == 2);

    // Initially, 0 filled, n empty
    SEM_INIT(&fill, 0);
    SEM_INIT(&empty, atoi(argv[1]));

    for (int i = 0; i < 8; i++) {
        create(T_produce);
        create(T_consume);
    }
}
```

## 哲学家就餐问题 ⭐️⭐️

- 下面的代码存在死锁

```c
void Tphilosopher(int id) {
    int lhs = (id + N - 1) % N;
    int rhs = id % N;

    while (1) {
        // Come to table
        // P(&table);

        P(&avail[lhs]);
        printf("+ %d by T%d\n", lhs, id);
        P(&avail[rhs]);
        printf("+ %d by T%d\n", rhs, id);

        // Eat.
        // Philosophers are allowed to eat in parallel.

        printf("- %d by T%d\n", lhs, id);
        printf("- %d by T%d\n", rhs, id);
        V(&avail[lhs]);
        V(&avail[rhs]);

        // Leave table
        // V(&table);
    }
}
```


## PIPE ⭐️⭐️⭐️

- named pipe
- anonymous pipe
- 推荐做课程实验 M5 `sperf`

对于 `named pipe`，我们可以通过 `mkfifo` 系统调用来创建一个命名管道，然后通过 `open` 来打开这个管道(读口采用 `O_RDONLY`，写口采用 `O_WRONLY`)，然后就可以通过 `read` 和 `write` 来进行读写操作。
示例代码：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

// We also have UNIX domain sockets for local inter-process
// communication--they also have a name in the file system
// like "/var/run/docker.sock". This is similar to a named
// pipe.
#define PIPE_NAME "/tmp/my_pipe"

void pipe_read() {
    int fd = open(PIPE_NAME, O_RDONLY);
    char buffer[1024];

    if (fd == -1) {
        perror("open");
        exit(1);
    }

    // Read from the pipe
    int num_read = read(fd, buffer, sizeof(buffer));
    if (num_read > 0) {
        printf("Received: %s\n", buffer);
    } else {
        printf("No data received.\n");
    }
    close(fd);
}

void pipe_write(const char *content) {
    // Open the pipe for writing
    int fd = open(PIPE_NAME, O_WRONLY);

    if (fd == -1) {
        perror("open");
        exit(1);
    }

    // Write the message to the pipe
    write(fd, content, strlen(content) + 1);
    close(fd);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s read|write [message]\n", argv[0]);
        return 1;
    }

    // Create the named pipe if it does not exist
    if (mkfifo(PIPE_NAME, 0666) == -1) {
        if (errno != EEXIST) {
            perror("mkfifo");
            return 1;
        }
    } else {
        printf("Created " PIPE_NAME "\n");
    }

    if (strcmp(argv[1], "read") == 0) {
        pipe_read();
    } else if (strcmp(argv[1], "write") == 0) {
        pipe_write(argv[2]);
    } else {
        fprintf(stderr, "Invalid command. Use 'read' or 'write'.\n");
        return 1;
    }

    return 0;
}
```

对于 `anonymous pipe`，可以通过 `pipe` 系统调用来创建一个匿名管道(读口写口参数为`size = 2`的数组 `int pipefd[2]`)，然后通过 `fork` 来创建一个子进程，通过 `close` 来关闭不需要的文件描述符，然后就可以通过 `read` 和 `write` 来进行读写操作。

- `dup2` 可以复制管道的读写端
- 文件描述符是一个用于访问文件或其他输入/输出资源的 “指针”
  
示例代码：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

// We also have UNIX domain sockets for local inter-process
// communication--they also have a name in the file system
// like "/var/run/docker.sock". This is similar to a named
// pipe.
#define PIPE_NAME "/tmp/my_pipe"

void pipe_read() {
    int fd = open(PIPE_NAME, O_RDONLY);
    char buffer[1024];

    if (fd == -1) {
        perror("open");
        exit(1);
    }

    // Read from the pipe
    int num_read = read(fd, buffer, sizeof(buffer));
    if (num_read > 0) {
        printf("Received: %s\n", buffer);
    } else {
        printf("No data received.\n");
    }
    close(fd);
}

void pipe_write(const char *content) {
    // Open the pipe for writing
    int fd = open(PIPE_NAME, O_WRONLY);

    if (fd == -1) {
        perror("open");
        exit(1);
    }

    // Write the message to the pipe
    write(fd, content, strlen(content) + 1);
    close(fd);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s read|write [message]\n", argv[0]);
        return 1;
    }

    // Create the named pipe if it does not exist
    if (mkfifo(PIPE_NAME, 0666) == -1) {
        if (errno != EEXIST) {
            perror("mkfifo");
            return 1;
        }
    } else {
        printf("Created " PIPE_NAME "\n");
    }

    if (strcmp(argv[1], "read") == 0) {
        pipe_read();
    } else if (strcmp(argv[1], "write") == 0) {
        pipe_write(argv[2]);
    } else {
        fprintf(stderr, "Invalid command. Use 'read' or 'write'.\n");
        return 1;
    }

    return 0;
}
```

## sh ⭐️⭐️⭐️

- 这个 Shell 没有引用任何库文件——它只通过系统调用访问操作系统中的对象。
- 下面为程序主要部分：

- 程序入口：

```c
void _start() {
    main();
    syscall(SYS_exit, 0);
}
```

- 结构体定义及主要命令处理函数：


```c
enum {
    EXEC = 1,
    REDIR,
    PIPE,
    LIST,
    BACK
};

#define MAXARGS 10
#define NULL ((void *)0)

struct cmd {
    int type;
};

struct execcmd {
    int type;
    char *argv[MAXARGS], *eargv[MAXARGS];
};

struct redircmd {
    int type, fd, mode;
    char *file, *efile;
    struct cmd *cmd;
};

struct pipecmd {
    int type;
    struct cmd *left, *right;
};

struct listcmd {
    int type;
    struct cmd *left, *right;
};

struct backcmd {
    int type;
    struct cmd *cmd;
};

struct cmd *parsecmd(char *);

// cmd is the "abstract syntax tree" (AST) of the command;
// runcmd() never returns.
void runcmd(struct cmd *cmd) {
    int p[2];
    struct backcmd *bcmd;
    struct execcmd *ecmd;
    struct listcmd *lcmd;
    struct pipecmd *pcmd;
    struct redircmd *rcmd;

    if (cmd == 0)
        syscall(SYS_exit, 1);

    switch (cmd->type) {
    case EXEC:
        ecmd = (struct execcmd *)cmd;
        if (ecmd->argv[0] == 0)
            syscall(SYS_exit, 1);

        char *c = zalloc(5 + strlen(ecmd->argv[0]) + 1);
        strcpy(c, "/bin/");
        strcpy(c + strlen(c), ecmd->argv[0]);
        syscall(SYS_execve, c, ecmd->argv, NULL);
        print("fail to exec ", c, "\n", NULL);
        break;

    case REDIR:
        rcmd = (struct redircmd *)cmd;
        syscall(SYS_close, rcmd->fd);
        if (syscall(SYS_open, rcmd->file, rcmd->mode, 0644) < 0) {
            print("fail to open ", rcmd->file, "\n", NULL);
            syscall(SYS_exit, 1);
        }
        runcmd(rcmd->cmd);
        break;

    case LIST:
        lcmd = (struct listcmd *)cmd;
        if (syscall(SYS_fork) == 0)
            runcmd(lcmd->left);
        syscall(SYS_wait4, -1, 0, 0, 0);
        runcmd(lcmd->right);
        break;

    case PIPE:
        pcmd = (struct pipecmd *)cmd;
        assert(syscall(SYS_pipe, p) >= 0);
        if (syscall(SYS_fork) == 0) {
            syscall(SYS_close, 1);
            syscall(SYS_dup, p[1]);
            syscall(SYS_close, p[0]);
            syscall(SYS_close, p[1]);
            runcmd(pcmd->left);
        }
        if (syscall(SYS_fork) == 0) {
            syscall(SYS_close, 0);
            syscall(SYS_dup, p[0]);
            syscall(SYS_close, p[0]);
            syscall(SYS_close, p[1]);
            runcmd(pcmd->right);
        }
        syscall(SYS_close, p[0]);
        syscall(SYS_close, p[1]);
        syscall(SYS_wait4, -1, 0, 0, 0);
        syscall(SYS_wait4, -1, 0, 0, 0);
        break;

    case BACK:
        bcmd = (struct backcmd *)cmd;
        if (syscall(SYS_fork) == 0)
            runcmd(bcmd->cmd);
        break;

    default:
        assert(0);
    }
    syscall(SYS_exit, 0);
}
```

可以看到将 `shell` 的命令分为了 `EXEC`、`REDIR`、`PIPE`、`LIST`、`BACK` 五种类型：
- `EXEC` 为执行命令
- `REDIR` 为重定向命令
- `PIPE` 为管道命令
- `LIST` 为列表命令
- `BACK` 为后台命令

重点看看对于 `PIPE` 的处理.
```c
// Author: Github Copilot
case PIPE:
    // 将cmd强制转换为pipecmd类型的指针，以便访问pipecmd特有的字段
    pcmd = (struct pipecmd *)cmd;
    // 创建一个管道，p[0]为读端，p[1]为写端。assert确保管道创建成功
    assert(syscall(SYS_pipe, p) >= 0);
    // 创建一个子进程
    if (syscall(SYS_fork) == 0) {
        // 在子进程中，关闭标准输出（文件描述符1）
        syscall(SYS_close, 1);
        // 将管道的写端复制到标准输出位置
        syscall(SYS_dup, p[1]);
        // 关闭管道的读端
        syscall(SYS_close, p[0]);
        // 关闭管道的写端（已经复制到标准输出，不再需要原始的文件描述符）
        syscall(SYS_close, p[1]);
        // 递归地执行管道左侧的命令
        runcmd(pcmd->left);
    }
    // 再次创建一个子进程
    if (syscall(SYS_fork) == 0) {
        // 在新的子进程中，关闭标准输入（文件描述符0）
        syscall(SYS_close, 0);
        // 将管道的读端复制到标准输入位置
        syscall(SYS_dup, p[0]);
        // 关闭管道的读端（已经复制到标准输入，不再需要原始的文件描述符）
        syscall(SYS_close, p[0]);
        // 关闭管道的写端
        syscall(SYS_close, p[1]);
        // 递归地执行管道右侧的命令
        runcmd(pcmd->right);
    }
    // 在父进程中，关闭管道的读端
    syscall(SYS_close, p[0]);
    // 在父进程中，关闭管道的写端
    syscall(SYS_close, p[1]);
    // 父进程等待第一个子进程完成
    syscall(SYS_wait4, -1, 0, 0, 0);
    // 父进程等待第二个子进程完成
    syscall(SYS_wait4, -1, 0, 0, 0);
    // 结束case语句
    break;
```

## dlbox ⭐️⭐️⭐️

- `dlbox` 是一个简单的动态链接库加载器，它可以加载并运行动态链接库中的函数。
- 对于其中所有的符号都采取"查表"的方式，而不是直接调用函数，这与 GOT(全局偏移表)的工作方式类似。

对于 `dl.h`:

```c
#ifdef __ASSEMBLER__

    #define DL_HEAD     __hdr: \
                        /* magic */    .ascii DL_MAGIC; \
                        /* file_sz */  .4byte (__end - __hdr); \
                        /* code_off */ .4byte (__code - __hdr)
    #define DL_CODE     .fill REC_SZ - 1, 1, 0; \
                        .align REC_SZ, 0; \
                        __code:
    #define DL_END      __end:

    #define RECORD(sym, off, name) \
        .align REC_SZ, 0; \
        sym .8byte (off); .ascii name

    #define IMPORT(sym) RECORD(sym:,           0, "?" #sym "\0")
    #define EXPORT(sym) RECORD(    , sym - __hdr, "#" #sym "\0")
    #define LOAD(lib)   RECORD(    ,           0, "+" lib  "\0")
    #define DSYM(sym)   *sym(%rip)

#else
    #include <stdint.h>

    struct dl_hdr {
        char magic[4];
        uint32_t file_sz, code_off;
    };

    struct symbol {
        int64_t offset;
        char type, name[REC_SZ - sizeof(int64_t) - 1];
    };
#endif
```

`dl.h` 对于直接汇编语言编写的代码定义了一些宏，如动态链接库的头部、代码段、结束标记、符号表记录等。对于 C 语言代码，定义了动态链接库头部和符号表记录的结构体。
- `DL_HEAD` 定义了动态链接库的头部，包括魔数、文件大小和代码段偏移量，在 `DL_HEAD` 与 `DL_CODE` 之间的部分写入符号表.
- `DL_CODE` 定义了代码段的起始位置，`DL_END` 定义了代码段的结束位置。
- `LOAD` 用于加载动态链接库，`IMPORT` 导入符号，`EXPORT` 导出符号，`DSYM` 调用符号(表示出了符号的地址)

如下面的汇编代码所示：
```c
#include "dl.h"

DL_HEAD

LOAD("libc.dl")
LOAD("libhello.dl")
IMPORT(hello)
IMPORT(exit)
EXPORT(_start)

DL_CODE

main:
    call DSYM(hello)
    call DSYM(hello)
    call DSYM(hello)
    call DSYM(hello)
    movq $0, %rax
    ret

_start:
    call main
    jmp DSYM(exit)

DL_END
```

定义了 `dl.h` 结合 `dlbox.c` 来使用：
```c
// 定义全局符号表和库表
static struct symbol *libs[16], syms[128];

// 根据符号名查找符号地址
static void *dlsym(const char *name);

// 将符号名和地址导出到全局符号表
static void dlexport(const char *name, void *addr);

// 加载一个库，如果库中的符号未加载则递归加载
static void dlload(struct symbol *sym);

// 打开并加载一个动态链接库文件，返回库的句柄
static struct dlib *dlopen(const char *path) {
    struct dl_hdr hdr; // 库头部信息
    struct dlib *h; // 库句柄

    // 打开库文件
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        goto bad;
    // 读取库头部信息
    if (read(fd, &hdr, sizeof(hdr)) < sizeof(hdr))
        goto bad;
    // 检查魔数是否匹配
    if (strncmp(hdr.magic, DL_MAGIC, strlen(DL_MAGIC)) != 0)
        goto bad;

    // 将库文件映射到内存
    h = mmap(NULL, hdr.file_sz, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE, fd, 0);
    if (h == MAP_FAILED)
        goto bad;

    // 设置库的符号表和路径
    h->symtab = (struct symbol *)((char *)h + REC_SZ);
    h->path = path;

    // 遍历符号表，根据符号类型进行处理
    for (struct symbol *sym = h->symtab; sym->type; sym++) {
        switch (sym->type) {
        case '+': // 递归加载依赖的库
            dlload(sym);
            break;
        case '?': // 解析符号地址
            sym->offset = (uintptr_t)dlsym(sym->name);
            break;
        case '#': // 导出符号
            dlexport(sym->name, (char *)h + sym->offset);
            break;
        }
    }

    return h;

bad:
    // 错误处理，关闭文件描述符并返回NULL
    if (fd > 0)
        close(fd);
    return NULL;
}

// 查找符号地址，如果找到返回地址，否则断言失败
static void *dlsym(const char *name) {
    for (int i = 0; i < LENGTH(syms); i++)
        if (strcmp(syms[i].name, name) == 0)
            return (void *)syms[i].offset; // 返回符号地址
    assert(0);
}

// 导出符号到全局符号表，如果表满则断言失败
static void dlexport(const char *name, void *addr) {
    for (int i = 0; i < LENGTH(syms); i++)
        if (!syms[i].name[0]) {
            syms[i].offset = (uintptr_t)addr; // 设置符号地址
            strcpy(syms[i].name, name); // 记录符号名
            return;
        }
    assert(0);
}

// 加载库，如果库已加载则返回，否则递归加载依赖的库
static void dlload(struct symbol *sym) {
    for (int i = 0; i < LENGTH(libs); i++) {
        if (libs[i] && strcmp(libs[i]->name, sym->name) == 0)
            return; // 库已加载
        if (!libs[i]) {
            libs[i] = sym;
            dlopen(sym->name); // 递归加载库
            return;
        }
    }
    assert(0); // 如果所有库槽位都已使用，则断言失败
}
```

## thread-os ⭐️⭐️⭐️
```c
#include <am.h>
#include <klib.h>
#include <klib-macros.h>

typedef union thread {
    struct {
        const char    *name;
        void          (*entry)(void *);
        Context       context;
        union thread  *next;
        char          end[0];
    };
    uint8_t stack[8192];
} Thread;

void T1(void *);
void T2(void *);
void T3(void *);

Thread threads[] = {
    // Context for the bootstrap code:
    { .name = "_", .entry = NULL, .next = &threads[1] },

    // Thread contests:
    { .name = "1", .entry = T1, .next = &threads[2] },
    { .name = "2", .entry = T2, .next = &threads[3] },
    { .name = "3", .entry = T3, .next = &threads[1] },
};
Thread *current = &threads[0];

Context *on_interrupt(Event ev, Context *ctx) {
    // Save context.
    current->context = *ctx;

    // Thread schedule.
    current = current->next;

    // Restore current thread's context.
    return &current->context;
}

int main() {
    cte_init(on_interrupt);

    for (int i = 1; i < LENGTH(threads); i++) {
        Thread *t = &threads[i];
        t->context = *kcontext(
            // a Thread object:
            // +--------------------------------------------+
            // | name, ... end[0] | Kernel stack ...        |
            // +------------------+-------------------------+
            // ^                  ^                         ^     
            // t                  &t->end                   t + 1
            (Area) { .start = &t->end, .end = t + 1, },
            t->entry, NULL
        );
    }

    yield();
    assert(0);  // Never returns.
}


void delay() {
    for (int volatile i = 0;
         i < 10000000; i++);
}

void T1(void *arg) { while (1) { putch('A'); delay(); } }
void T2(void *arg) { while (1) { putch('B'); delay(); } }
void T3(void *arg) { while (1) { putch('C'); delay(); } }
```


- 这是一个最简易的 `os` 模型.
- 通过 `on_interrupt` 函数实现了线程的调度. 
- 借助 `abstract machine` 的 `Context` 结构体以及 `yield` 函数实现了线程的切换. `AM` 项目在 `NJU ProjectN` 项目中可以找到.


## xv6-riscv ⭐️⭐️⭐️⭐️⭐️

- 特别精简但完整的操作系统实现，适合读源码学习.
- 很重要.

Makefile 下载 xv6-riscv 项目并提供了 python 调试脚本

```makefile
xv6-riscv:
	git clone https://github.com/mit-pdos/xv6-riscv.git

debug:
	gdb-multiarch -x init.py
```

```python
import gdb
import re

R = {}

def stop_handler(event):
    if isinstance(event, gdb.StopEvent):
        regs = [
            line for line in 
                gdb.execute('info registers',
                            to_string=True).
                            strip().split('\n')
        ]
        for line in regs:
            parts = line.split()
            key = parts[0]

            if m := re.search(r'(\[.*?\])', line):
                val = m.group(1)
            else:
                val = parts[1]

            if key in R and R[key] != val:
                print(key, R[key], '->', val)
            R[key] = val

gdb.events.stop.connect(stop_handler)

gdb.execute('set confirm off')
gdb.execute('set architecture riscv:rv64')
gdb.execute('target remote 127.0.0.1:26000')
gdb.execute('symbol-file xv6-riscv/kernel/kernel')
gdb.execute('set riscv use-compressed-breakpoints yes')

# Set a breakpoint on trampoline
# All user traps go here.
gdb.execute('hb *0x3ffffff000')

# User program entry
gdb.execute('hb *0x0')
```

- To be continued...(尚未认真学习 `xv6-riscv` 项目)



## launcher ⭐️⭐️⭐️

- 设备驱动程序
> Everything is a file.

- 实现设备驱动程序本质上就是提供一组文件操作接口(`read`,`write`...)，这些接口可以被用户态程序调用，从而实现对设备的控制。

即最重要的是实现下面的接口：
```c
static ssize_t launcher_read(struct file *, char __user *, size_t, loff_t *);
static ssize_t launcher_write(struct file *, const char __user *, size_t, loff_t *);
```

完整代码：


```c
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/uaccess.h>

#define NUM_DEV 2

static int dev_major = 0;
static struct class *launcher_class = NULL;
static struct cdev cdev;

static ssize_t launcher_read(struct file *, char __user *, size_t, loff_t *);
static ssize_t launcher_write(struct file *, const char __user *, size_t, loff_t *);

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .read = launcher_read,
    .write = launcher_write,
};

static struct nuke {
    struct cdev cdev;
} devs[NUM_DEV];

static int __init launcher_init(void) {
    int i;
    dev_t dev;

    // allocate device range
    alloc_chrdev_region(&dev, 0, 1, "nuke");

    // create device major number
    dev_major = MAJOR(dev);

    // create class
    launcher_class = class_create(THIS_MODULE, "nuke");
    cdev.owner = THIS_MODULE;

    for (i = 0; i < NUM_DEV; i++) {
        // register device
        cdev_init(&devs[i].cdev, &fops);
        cdev_add(&devs[i].cdev, MKDEV(dev_major, i), 1);
        device_create(launcher_class, NULL, MKDEV(dev_major, i), NULL, "nuke%d", i);
    }
    return 0;
}

static void __exit launcher_exit(void) {
    device_destroy(launcher_class, MKDEV(dev_major, 0));
    unregister_chrdev_region(MKDEV(dev_major, 0), MINORMASK);
    class_unregister(launcher_class);
    class_destroy(launcher_class);
}

static ssize_t launcher_read(struct file *file, char __user *buf, size_t count, loff_t *offset) {
    if (*offset != 0) {
        return 0;
    } else {
        uint8_t *data = "This is dangerous!\n";
        size_t datalen = strlen(data);
        if (count > datalen) {
          count = datalen;
        }
        if (copy_to_user(buf, data, count)) {
          return -EFAULT;
        }
        *offset += count;
        return count;
    }
}

static ssize_t launcher_write(struct file *file, const char __user *buf, size_t count, loff_t *offset) {
    char databuf[4] = "\0\0\0\0";
    if (count > 4) {
        count = 4;
    }

    copy_from_user(databuf, buf, count);
    if (strncmp(databuf, "\x01\x14\x05\x14", 4) == 0) {
        const char *EXPLODE[] = {
          "    ⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣀⣀⠀⠀⣀⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
          "    ⠀⠀⠀⣀⣠⣤⣤⣾⣿⣿⣿⣿⣷⣾⣿⣿⣿⣿⣿⣶⣿⣿⣿⣶⣤⡀⠀⠀⠀⠀",
          "    ⠀⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀",
          "    ⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⡀⠀",
          "    ⠀⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀",
          "    ⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠁⠀",
          "    ⠀⠀⠻⢿⡿⢿⣿⣿⣿⣿⠟⠛⠛⠋⣀⣀⠙⠻⠿⠿⠋⠻⢿⣿⣿⠟⠀⠀⠀⠀",
          "    ⠀⠀⠀⠀⠀⠀⠈⠉⣉⣠⣴⣷⣶⣿⣿⣿⣿⣶⣶⣶⣾⣶⠀⠀⠀⠀⠀⠀⠀⠀",
          "    ⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠋⠈⠛⠿⠟⠉⠻⠿⠋⠉⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀",
          "    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣶⣷⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
          "    ⠀⠀⠀⠀⠀⠀⢀⣀⣠⣤⣤⣤⣤⣶⣿⣿⣷⣦⣤⣤⣤⣤⣀⣀⠀⠀⠀⠀⠀⠀",
          "    ⠀⠀⠀⠀⢰⣿⠛⠉⠉⠁⠀⠀⠀⢸⣿⣿⣧⠀⠀⠀⠀⠉⠉⠙⢻⣷⠀⠀⠀⠀",
          "    ⠀⠀⠀⠀⠀⠙⠻⠷⠶⣶⣤⣤⣤⣿⣿⣿⣿⣦⣤⣤⣴⡶⠶⠟⠛⠁⠀⠀⠀⠀",
          "    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
          "    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠒⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠓⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        };
        int i;

        for (i = 0; i < sizeof(EXPLODE) / sizeof(EXPLODE[0]); i++) {
          printk("\033[01;31m%s\033[0m\n", EXPLODE[i]);
      }
    } else {
      printk("nuke: incorrect secret, cannot lanuch.\n");
    }
    return count;
}

module_init(launcher_init);
module_exit(launcher_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("jyy");
```


## readfat ⭐️⭐️⭐️⭐️⭐️⭐️

- To Be Continued (目前尚未学习该文档)
- 具体最好详细阅读 [Microsoft FAT Specification](https://jyywiki.cn/OS/manuals/MSFAT-spec.pdf)
- 文档和该项目代码可以对照阅读，会有新的体验

- fat32.h

```c
#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;

// Copied from the manual

struct fat32hdr {
    u8  BS_jmpBoot[3];
    u8  BS_OEMName[8];
    u16 BPB_BytsPerSec;
    u8  BPB_SecPerClus;
    u16 BPB_RsvdSecCnt;
    u8  BPB_NumFATs;
    u16 BPB_RootEntCnt;
    u16 BPB_TotSec16;
    u8  BPB_Media;
    u16 BPB_FATSz16;
    u16 BPB_SecPerTrk;
    u16 BPB_NumHeads;
    u32 BPB_HiddSec;
    u32 BPB_TotSec32;
    u32 BPB_FATSz32;
    u16 BPB_ExtFlags;
    u16 BPB_FSVer;
    u32 BPB_RootClus;
    u16 BPB_FSInfo;
    u16 BPB_BkBootSec;
    u8  BPB_Reserved[12];
    u8  BS_DrvNum;
    u8  BS_Reserved1;
    u8  BS_BootSig;
    u32 BS_VolID;
    u8  BS_VolLab[11];
    u8  BS_FilSysType[8];
    u8  __padding_1[420];
    u16 Signature_word;
} __attribute__((packed));

struct fat32dent {
    u8  DIR_Name[11];
    u8  DIR_Attr;
    u8  DIR_NTRes;
    u8  DIR_CrtTimeTenth;
    u16 DIR_CrtTime;
    u16 DIR_CrtDate;
    u16 DIR_LastAccDate;
    u16 DIR_FstClusHI;
    u16 DIR_WrtTime;
    u16 DIR_WrtDate;
    u16 DIR_FstClusLO;
    u32 DIR_FileSize;
} __attribute__((packed));

#define CLUS_INVALID   0xffffff7

#define ATTR_READ_ONLY 0x01
#define ATTR_HIDDEN    0x02
#define ATTR_SYSTEM    0x04
#define ATTR_VOLUME_ID 0x08
#define ATTR_DIRECTORY 0x10
#define ATTR_ARCHIVE   0x20

```

- readfat.c


```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fcntl.h>  
#include <unistd.h>
#include <sys/mman.h>
#include "fat32.h"

struct fat32hdr *hdr;

void *mmap_disk(const char *fname);
void dfs_scan(u32 clusId, int depth, int is_dir);

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s fs-image\n", argv[0]);
        exit(1);
    }

    setbuf(stdout, NULL);

    assert(sizeof(struct fat32hdr) == 512);
    assert(sizeof(struct fat32dent) == 32);

    // Map disk image to memory.
    // The file system is a in-memory data structure now.
    hdr = mmap_disk(argv[1]);

    // File system traversal.
    dfs_scan(hdr->BPB_RootClus, 0, 1);

    munmap(hdr, hdr->BPB_TotSec32 * hdr->BPB_BytsPerSec);
}

void *mmap_disk(const char *fname) {
    int fd = open(fname, O_RDWR);

    if (fd < 0) {
        goto release;
    }

    off_t size = lseek(fd, 0, SEEK_END);
    if (size < 0) {
        goto release;
    }

    struct fat32hdr *hdr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (hdr == MAP_FAILED) {
        goto release;
    }

    close(fd);

    assert(hdr->Signature_word == 0xaa55); // this is an MBR
    assert(hdr->BPB_TotSec32 * hdr->BPB_BytsPerSec == size);

    printf("%s: DOS/MBR boot sector, ", fname);
    printf("OEM-ID \"%s\", ", hdr->BS_OEMName);
    printf("sectors/cluster %d, ", hdr->BPB_SecPerClus);
    printf("sectors %d, ", hdr->BPB_TotSec32);
    printf("sectors %d, ", hdr->BPB_TotSec32);
    printf("sectors/FAT %d, ", hdr->BPB_FATSz32);
    printf("serial number 0x%x\n", hdr->BS_VolID);
    return hdr;

release:
    perror("map disk");
    if (fd > 0) {
        close(fd);
    }
    exit(1);
}

u32 next_cluster(int n) {
    // RTFM: Sec 4.1

    u32 off = hdr->BPB_RsvdSecCnt * hdr->BPB_BytsPerSec;
    u32 *fat = (u32 *)((u8 *)hdr + off);
    return fat[n];
}

void *cluster_to_sec(int n) {
    // RTFM: Sec 3.5 and 4 (TRICKY)
    // Don't copy code. Write your own.

    u32 DataSec = hdr->BPB_RsvdSecCnt + hdr->BPB_NumFATs * hdr->BPB_FATSz32;
    DataSec += (n - 2) * hdr->BPB_SecPerClus;
    return ((char *)hdr) + DataSec * hdr->BPB_BytsPerSec;
}

void get_filename(struct fat32dent *dent, char *buf) {
    // RTFM: Sec 6.1

    int len = 0;
    for (int i = 0; i < sizeof(dent->DIR_Name); i++) {
        if (dent->DIR_Name[i] != ' ') {
            if (i == 8)
                buf[len++] = '.';
            buf[len++] = dent->DIR_Name[i];
        }
    }
    buf[len] = '\0';
}

void dfs_scan(u32 clusId, int depth, int is_dir) {
    // RTFM: Sec 6

    for (; clusId < CLUS_INVALID; clusId = next_cluster(clusId)) {

        if (is_dir) {
            int ndents = hdr->BPB_BytsPerSec * hdr->BPB_SecPerClus / sizeof(struct fat32dent);

            for (int d = 0; d < ndents; d++) {
                struct fat32dent *dent = (struct fat32dent *)cluster_to_sec(clusId) + d;
                if (dent->DIR_Name[0] == 0x00 ||
                    dent->DIR_Name[0] == 0xe5 ||
                    dent->DIR_Attr & ATTR_HIDDEN)
                    continue;

                char fname[32];
                get_filename(dent, fname);

                for (int i = 0; i < 4 * depth; i++)
                    putchar(' ');
                printf("[%-12s] %6.1lf KiB    ", fname, dent->DIR_FileSize / 1024.0);

                u32 dataClus = dent->DIR_FstClusLO | (dent->DIR_FstClusHI << 16);
                if (dent->DIR_Attr & ATTR_DIRECTORY) {
                    printf("\n");
                    if (dent->DIR_Name[0] != '.') {
                        dfs_scan(dataClus, depth + 1, 1);
                    }
                } else {
                    dfs_scan(dataClus, depth + 1, 0);
                    printf("\n");
                }
            }
        } else {
            printf("#%d ", clusId);
        }
    }
}

```

>>>>>>> master
