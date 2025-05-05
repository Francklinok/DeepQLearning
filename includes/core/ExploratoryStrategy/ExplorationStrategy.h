#pragma once

class ExplorationStrategy {
public:
	virtual ~ExplorationStrategy() = default;
	virtual float getExploratoryRate(int step) const = 0;
	virtual void reset() = 0;


};