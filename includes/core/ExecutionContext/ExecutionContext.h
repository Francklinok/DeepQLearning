template <typename T = float>

class ExecutionContext {
public:
	ExecutionContext(bool enableParallel = true, int  threads = -1);
	T generateNormalRandom();
	T generateUniformRandom();
	bool  isParalleleEnabled() const;
	T getCachedValue(int64_t step, const  std::string& key);
	void  setCachedValue(int64_t step, const std::string& key, T value) const;
	void  clearCache() const;

private:
	bool   parallelExecution;
	int  numThreads;
	mutable std::mutex randomMutex;
	std::mt19937 randomEngine;
	std::normal_distribution<double> normalDist;
	mutable std::shared_lutex  cacheMutex;
	mutable  std::unordered_map<std::pair<int64_t, std::string>, T, , PairHash<int64_t, std::string>>  computeCache;
	//hash fonction

	template  <typename A ,  typename B>
	struct  PairHash {
		std::size_t  operator()(const std::pair<A, B>& p) const {
			auto  h1 = std::hash<A>{}(p.first);
			auto h2 = std::hash<B>{}(p.second);
			return  h1 ^ (h2 << 1);
		}
	};

};
