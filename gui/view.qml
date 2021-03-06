import QtQuick 2.0
import QtQuick.Layouts 1.12
import QtQuick.Controls 2.12
import QtQuick.Window 2.12


ApplicationWindow {
    id: page
    width: 800
    height: 400
    visible: true
    GridLayout {
        id: grid
        columns: 2
        rows: 1
        height:parent.height
        width:parent.width
        ColumnLayout {
            spacing: 2
            Layout.preferredWidth: 100
            Layout.fillHeight: true 
            Layout.fillWidth: true
            Column {
                Button {
                    text: "Push"
                    onClicked: stacklayout.currentIndex = 0
                }            
            }      
        }
        ColumnLayout {
            Layout.preferredWidth: 700
            Layout.fillHeight: true 
            StackLayout {
                id: stacklayout
                currentIndex: 1
                Layout.fillWidth: true  
                Rectangle {
                    color: 'teal'
                    Layout.fillWidth: true  
                    implicitWidth: 200
                    implicitHeight: 200
                }

                TextArea {
                    id: multiline
                    Layout.fillWidth: true  
                    placeholderText: "Initial text\n...\n...\n"
                    background: Rectangle {
                       implicitWidth: 200
                       implicitHeight: 100
                       border.color: multiline.focus ? "#21be2b" : "lightgray"
                       color: multiline.focus ? "lightgray" : "transparent"
                    }
                }                     

            }

        }
    }   

}