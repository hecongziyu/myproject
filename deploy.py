# -*- coding:utf-8 -*-
import paramiko
import argparse
from sshtunnel import SSHTunnelForwarder
import os
import sys
class DeployClient(object):
    def __init__(self,client_ip, port, user_name, password):
        self.client_ip = client_ip
        self.user_name = user_name
        self.password = password
        self.sshclient = paramiko.SSHClient() 
        self.sshclient.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.sshclient.connect(client_ip,port,user_name,password)
        print('deploy connected {} {}'.format(client_ip, port))

    def execute(self,command):
        stdin, stdout, stderr = self.sshclient.exec_command(command)
        result = stdout.readlines()
        error = stderr.readlines()
        return result, error

    def __del__(self):
        self.sshclient.close()


class DeploySSHTunel(object):
    def __init__(self, client_ip, port, user_name, password, 
                        remote_ip, remote_port, remote_user_name, remote_password):
        self.client_ip = client_ip
        self.port = port
        self.user_name = user_name
        self.password = password
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.remote_user_name = remote_user_name
        self.remote_password = remote_password
        self.server = None
        # self.__init_ssh_tunel__()
        # Deploy.__init__('127.0.0.1', self.server.local_bind_port, remote_user_name, remote_password)

    def start(self):
        self.server = SSHTunnelForwarder(
            (self.client_ip,self.port),
            ssh_username=self.user_name,
            ssh_password=self.password,
            remote_bind_address=(self.remote_ip, self.remote_port),
			local_bind_address=('127.0.0.1', 10022))        
        self.server.start()
      

    def __del__(self):
        if self.server:
            self.server.stop()
        
    def get_deploy(self):
        print('local bind port {}'.format(self.server.local_bind_port))
        client = DeployClient('127.0.0.1', self.server.local_bind_port, self.remote_user_name, self.remote_password)
        return client

    def stop(self):
        if self.server:
            self.server.stop()



if __name__ == '__main__':			
    print('local listen port 10022')
    command = sys.argv[1]
    result = 0
    print(command)
    if command:
        try:
            server = DeploySSHTunel('139.159.209.132',18922,'cloud','talkweb!@#','192.168.1.10',2376,'','')
            server.start()
            
            server.stop()
        except:
            server.stop()
    sys.exit(result)
