import {rankParameters} from './discovery.js';

self.onmessage = event => {
  try {
    self.postMessage({result:rankParameters(event.data.trials,event.data.metric)});
  } catch (error) {
    self.postMessage({error:error.message});
  }
};
