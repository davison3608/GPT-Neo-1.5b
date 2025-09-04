import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Controls 2.12

Window {
    id: root
    visible: true
    width: 300
    height: 100
    title: "Assist Terminal"
    // 设置透明度属性
    opacity: 0.8
    flags: Qt.Window | Qt.WindowStaysOnTopHint

    // 直角半透明紫色背景面板
    Rectangle {
        id: background
        anchors.fill: parent
        color: "#30152B"  // 紫色带透明度
        border.color: "#30152B"
        border.width: 1
        // 确保背景在最底层
        z: -1
    }

    // 主内容列布局
    Column {
        anchors.centerIn: parent
        spacing: 20

        // 第一个输入行 - 使用slWriter
        Row
        {
            spacing: 10
            height: 40

            TextField {
                id: sql_find
                width: 200
                height: 30
                placeholderText: "find_or_down"
                background: Rectangle {
                    color: "#f0f0f0"
                    border.color: "#d3d3d3"
                }
            }

            Button {
                width: 60
                height: 30
                text: "sub"
                background: Rectangle {
                    color: parent.down ? "#30152B" : "#f0f0f0"
                }
                onClicked: {
                    console.log("sql_find: ", sql_find.text)
                    // 调用slWriter的write_to方法，传入输入框文本
                    slWriter.write_to(sql_find.text)
                }
            }
        }

        // 第二个输入行 - 使用luWriter
        Row
        {
            spacing: 10
            height: 40

            TextField {
                id: lru_find
                width: 200
                height: 30
                placeholderText: "find_or_certain_key"
                background: Rectangle {
                    color: "#f0f0f0"
                    border.color: "#d3d3d3"
                }
            }

            Button {
                width: 60
                height: 30
                text: "sub"
                background: Rectangle {
                    color: parent.down ? "#30152B" : "#f0f0f0"
                }
                onClicked: {
                    console.log("lru_find: ", lru_find.text)
                    // 调用luWriter的write_to方法，传入输入框文本
                    luWriter.write_to(lru_find.text)
                }
            }
        }
    }
}
    