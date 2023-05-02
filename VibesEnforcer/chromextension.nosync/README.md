## Sunflower Daylight Extension

This extension rewrites all tweets with an LLM (GPT-3.5) and is intended to reduce cognitohazards and generally make the experience of Twitter more pleasant.

## Installation

1. Download this folder from the repo:
```
git clone --depth 1 --no-checkout https://github.com/cameronfr/SunflowerOS.git
cd SunflowerOS
git sparse-checkout set VibesEnforcer/chromextension.nosync
git checkout master
cd VibesEnforcer/chromextension.nosync
```

2. Set your OpenAI API key.
```
vim +19 src/pages/Content/index.jsx
# Set your OpenAI API key on line 19
# You can also customize the prompt here (mine is pretty good though. LMK if you find better ones!!)
```

3. Build the extension.
```
npm install
npm run build
```

4. Install the extension.
- In Chrome, go to `chrome://extensions/`, enable developer mode, and click "Load unpacked". Select the `SunflowerOS/VibesEnforcer/chromextension.nosync/build` folder.

Please message me on the Sunflower [Discord](https://discord.com/invite/zYmm5JuHkW) if you have any questions or issues, or if you have any ideas and want to build something similar!