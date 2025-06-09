#pragma once
#include <memory>
#include <vector>
#include <array>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <type_traits>
#include <concepts>
#include <immintrin.h>
#include <execution>
#include <algorithm>
#include <numeric>

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{ false };

    static constexpr size_t CACHE_LINE_SIZE = 64;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> active_tasks{ 0 };

public:
    explicit ThreadPool(size_t threads = std::thread::hardware_concurrency() * 2) {
        workers.reserve(threads);
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    active_tasks.fetch_add(1, std::memory_order_relaxed);
                    task();
                    active_tasks.fetch_sub(1, std::memory_order_relaxed);
                }
                });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    void wait_all() {
        while (active_tasks.load(std::memory_order_relaxed) > 0) {
            std::this_thread::yield();
        }
    }

    ~ThreadPool() {
        stop = true;
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }
};
