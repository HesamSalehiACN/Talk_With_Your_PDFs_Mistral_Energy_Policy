css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #FFFFFF
}
.chat-message.bot {
    background-color: #F5F5F5
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 50px;
  max-height: 50px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #000;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.freeiconspng.com/uploads/gear-icon-13.png" style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.freeiconspng.com/uploads/people-icon--icon-search-engine-18.png" style="max-height: 50px; max-width: 50px;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
