#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <array>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <atomic>
#include <type_traits>
#include <concepts>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace deep_qn {
    namespace core {

        enum class TaskPriority {
            Low = 0, Normal = 1, High = 2, Critical = 3
        };

        struct Task {
            std::function<void()> function;
            TaskPriority priority;
            std::chrono::steady_clock::time_point submit_time;

            Task(std::function<void()> f, TaskPriority p = TaskPriority::Normal)
                : function(std::move(f)), priority(p), submit_time(std::chrono::steady_clock::now()) {
            }

            bool operator<(const Task& other) const {
                if (priority != other.priority)
                    return priority < other.priority;
                return submit_time > other.submit_time;
            }
        };

        struct ThreadPoolStats {
            std::atomic<size_t> tasks_submitted{ 0 };
            std::atomic<size_t> tasks_completed{ 0 };
            std::atomic<size_t> tasks_failed{ 0 };
            std::atomic<size_t> current_queue_size{ 0 };
            std::atomic<size_t> peak_queue_size{ 0 };
            std::atomic<size_t> active_threads{ 0 };
            std::atomic<double> average_task_time{ 0.0 };
        };

        struct ThreadPoolConfig {
            size_t num_threads = std::thread::hardware_concurrency() * 2;
            size_t max_queue_size = 10000;
            bool enable_priority_queue = true;
            bool enable_statistics = true;
        };

        class ThreadPool {
        private:
            std::vector<std::thread> workers;
            std::priority_queue<Task> task_queue;
            std::queue<Task> simple_queue;

            std::mutex queue_mutex;
            std::condition_variable condition;
            std::atomic<bool> stop{ false };
            std::atomic<size_t> active_tasks{ 0 };

            ThreadPoolConfig config;
            ThreadPoolStats stats;

            void worker_loop() {
                for (;;) {
                    Task task([] {}, TaskPriority::Normal);

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !task_queue.empty() || !simple_queue.empty();
                            });

                        if (stop && task_queue.empty() && simple_queue.empty()) return;

                        if (config.enable_priority_queue && !task_queue.empty()) {
                            task = std::move(const_cast<Task&>(task_queue.top()));
                            task_queue.pop();
                        }
                        else if (!config.enable_priority_queue && !simple_queue.empty()) {
                            task = std::move(simple_queue.front());
                            simple_queue.pop();
                        }
                        else continue;
                    }

                    if (config.enable_statistics) {
                        stats.active_threads++;
                    }

                    auto start = std::chrono::steady_clock::now();
                    try {
                        task.function();
                        if (config.enable_statistics) stats.tasks_completed++;
                    }
                    catch (...) {
                        if (config.enable_statistics) stats.tasks_failed++;
                    }

                    auto end = std::chrono::steady_clock::now();
                    if (config.enable_statistics) {
                        double duration = std::chrono::duration<double>(end - start).count();
                        stats.average_task_time = (stats.average_task_time.load() + duration) / 2.0;
                        stats.active_threads--;
                    }
                }
            }

        public:
            explicit ThreadPool(const ThreadPoolConfig& cfg = ThreadPoolConfig{}) : config(cfg) {
                workers.reserve(config.num_threads);
                for (size_t i = 0; i < config.num_threads; ++i)
                    workers.emplace_back([this] { worker_loop(); });
            }

            ~ThreadPool() {
                stop = true;
                condition.notify_all();
                for (auto& w : workers) w.join();
            }

            template<class F, class... Args>
            auto enqueue(F&& f, Args&&... args)
                -> std::future<typename std::result_of<F(Args...)>::type> {
                return enqueue_priority(TaskPriority::Normal, std::forward<F>(f), std::forward<Args>(args)...);
            }

            template<class F, class... Args>
            auto enqueue_priority(TaskPriority priority, F&& f, Args&&... args)
                -> std::future<typename std::result_of<F(Args...)>::type> {
                using return_type = typename std::result_of<F(Args...)>::type;

                auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                );
                std::future<return_type> res = task_ptr->get_future();

                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");

                    Task task([task_ptr]() { (*task_ptr)(); }, priority);

                    if (config.enable_priority_queue)
                        task_queue.push(std::move(task));
                    else
                        simple_queue.push(std::move(task));

                    if (config.enable_statistics) {
                        stats.tasks_submitted++;
                        stats.current_queue_size = (config.enable_priority_queue ? task_queue.size() : simple_queue.size());
                        if (stats.current_queue_size > stats.peak_queue_size)
                            stats.peak_queue_size = stats.current_queue_size;
                    }
                }

                condition.notify_one();
                return res;
            }

            void wait_all() {
                while (active_tasks.load(std::memory_order_relaxed) > 0)
                    std::this_thread::yield();
            }

            ThreadPoolStats get_statistics() const { return stats; }
        };

    } // namespace core
} // namespace deep_qn
