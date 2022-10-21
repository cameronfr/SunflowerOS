Modification of [Moonlight Embedded](https://github.com/moonlight-stream/moonlight-embedded/tree/f021439d1bb33b4869273f7521ec77edb6804fe1), only `src/` files have been modified (not `libgamestream`).

Steps 
1. Goto moonlight embedded repo, download `libgamestream` and `third_party` folders, add them to the root.
  - https://github.com/moonlight-stream/moonlight-embedded/tree/f021439d1bb33b4869273f7521ec77edb6804fe1
2. `cmake . && make -j8`
3. See `moonlightsdk.h` for the interface. 


