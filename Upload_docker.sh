docker build . -t  tengx/open_instruct_main_google2

beaker image delete  tengx/open_instruct_main_google2

beaker image create tengx/open_instruct_main_google2 --name open_instruct_main_google2