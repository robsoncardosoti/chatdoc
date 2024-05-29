css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 51px;
  max-height: 51px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.tcmgo.tc.br/pentaho/api/repos/cidadao/static/libs/img/tcmgo-robot.png" style="max-height: 51px; max-width: 51px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.tcmgo.tc.br/pentaho/api/repos/cidadao/static/libs/img/tcmgo-user.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

#head_template = """
#        <style>
#            div[data-testid="column"]:nth-of-type(1)
#            {text-align: end;}
#            div[data-testid="column"]:nth-of-type(2)
#            {text-align: begin;color: red;}
#            div[data-testid="column"]:nth-of-type(3)
#            {text-align: end;color: gray;}
#        </style>
#"""

logo_image = '''
<div>
    <img src="https://www.tcmgo.tc.br/site/wp-content/uploads/2017/06/logo_footer-1.png">
</div>
'''

