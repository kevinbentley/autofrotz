# autofrotz
AI that can play Frotz games

This currently uses dfrotz and stdin/stdout pipes to play. I found a python library after I already made the piping work, so I might convert it later.

You can install frotz with
```
sudo apt install frotz
```
You need the following evironment variables set to make this work:

```
ANTHROPIC_API_KEY
ELEVENLABS_API_KEY
```

If you are using Slack:
```
SLACK_WEBHOOK_URL
```

Game files come from https://github.com/danielricks/textplayer.git

```
git submodule init
git submodule update
```