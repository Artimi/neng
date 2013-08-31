# Neng #

Nash equilibria noncooperative games.

This tool serves to computing [Nash equilibrium](http://en.wikipedia.org/wiki/Nash_equilibrium) in
[noncooperative games](http://en.wikipedia.org/wiki/Non-cooperative_game). 
## Installation ##

    python setup.py install

## Usage ##

### Command-line ###

You can invoke the Nash equilibria computing with this parameters:
```
usage: game.py [-h] -f FILE
               [-m {L-BFGS-B,SLSQP,CMAES,support_enumeration,pne}] [-e] [-p]
               [-c] [-t {normalization,penalization}]
               [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--log-file LOG_FILE]

NenG - Nash Equilibrium Noncooperative Games.
Tool for computing Nash equilibria in noncooperative games.
Specifically:
All pure Nash equilibria in all games (--method=pne).
All mixed Nash equilibria in two-players games (--method=support_enumeration).
One sample mixed Nash equilibria in n-players games (--method={CMAES,L-BFGS-B,SLSQP}).

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  File where game in nfg format is saved.
  -m {L-BFGS-B,SLSQP,CMAES,support_enumeration,pne}, --method {L-BFGS-B,SLSQP,CMAES,support_enumeration,pne}
                        Method to use for computing Nash equlibria.
  -e, --elimination     Use Iterative Elimination of Strictly Dominated
                        Strategies before computing NE.
  -p, --payoff          Print also players payoff with each Nash equilibrium.
  -c, --checkNE         After computation check if found strategy profile is
                        really Nash equilibrium.
  -t {normalization,penalization}, --trim {normalization,penalization}
                        Method for keeping strategy profile in probability
                        distribution universum.
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Level of logs to save/print
  --log-file LOG_FILE   Log file. If omitted log is printed to stdout.
```

File format comes from [gambit file formats](http://www.gambit-project.org/doc/formats.html)

### Python API ###
You can find current API documentation at [readthedocs](https://neng.readthedocs.org/en/latest/).

### Travis CI ###
Every commit neng is built and tested using Travis CI. You can see result at [neng travis page](https://travis-ci.org/Artimi/neng).

## Bachelor thesis ##
This program result from my [bachelor thesis](https://www.fit.vutbr.cz/study/DP/BP.php?id=15471&y=2012) (in czech). In thesis are described basic math rules of Game theory and algorithms used in the program. It also contains some experiments and testing. But naturally development is moving forward and thesis is becoming obsolete with every commit.

## License ##
This program is licensed under MIT license.
