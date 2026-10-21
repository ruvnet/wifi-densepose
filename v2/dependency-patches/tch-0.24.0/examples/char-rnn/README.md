This example implements a character-level language model heavily inspired 
by [char-rnn](https://github.com/karpathy/char-rnn).
- For training, the model takes as input a text file and learns to predict the
  next character following a given sequence.
- For generation, the model uses a fixed seed, returns the next character
  distribution from which a character is randomly sampled and this is iterated
  to generate a text.

At the end of each training epoch, some sample text is generated and printed.

Any text file can be used as an input, as long as it's large enough for training.
A typical example would be the
[tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
The training text file should be stored in `data/input.txt`.

Compiling and running the example is done via the following command lines:
```bash
cargo run --example char-rnn
```

Here is an example of generated data when training on the Shakespeare dataset after only 5 epochs.
```
CAMILLO:
Nurse! Wast thou, to eat to be proud it.
My lords, he's gone; hers' falsehsof with a temper,
And all these silk of this issuel, or puck
And the new-vangance is danlike to take and turn;
Let him be through my brother and your arries
It sets do so, a knowledge itself have done!

HENRY BOLINGBROKE:
Come, come, when, sir, along is war,
Horsein your downrial his worshiph, fair sprenched
feignaconal take for thee, dispatch'd with living
Ingracious enching you. Then, great duke
Can klecked found my propter heart, and fees?

NORTHUMBERLAND:
Have youths of face, I know, that living sons
Through us within the house of the dear inn:
Before thy fire and loving eyes,--
```

