import {rankParameters} from './discovery.js';
import {setMeasures} from './parameters.js';

self.onmessage = event => {
  try {
    setMeasures(event.data.measures);
    self.postMessage({result:rankParameters(event.data.trials,event.data.metric)});
  } catch (error) {
    self.postMessage({error:error.message});
  }
};
